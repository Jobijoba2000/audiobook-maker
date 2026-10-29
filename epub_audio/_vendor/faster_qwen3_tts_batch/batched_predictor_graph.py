#!/usr/bin/env python3
"""
Batched CUDA graph capture for the code predictor's 15-step decode loop.

Extends PredictorGraph to support fixed max_batch_size. All static buffers
have shape [B, ...] and the entire 15-step loop is captured as one CUDA graph.

Key insight: the predictor is stateless (cache reset each call), so all B samples
share the same cache positions and attention masks. Only input/output tensors differ.
"""
import torch
from transformers import StaticCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

from .batched_sampling import batched_sample_logits


class BatchedPredictorGraph:
    """
    Batched version of PredictorGraph.

    Captures the full 15-step predictor loop for max_batch_size samples
    as a single CUDA graph.

    Usage:
        bpg = BatchedPredictorGraph(code_predictor, pred_config, talker_hidden_size, max_batch_size=4)
        bpg.capture()
        codebook_tokens = bpg.run(pred_input)  # pred_input: [B, 2, H] -> [B, 15]
    """

    def __init__(
        self,
        code_predictor,
        pred_config,
        talker_hidden_size,
        max_batch_size=4,
        device='cuda',
        dtype=torch.bfloat16,
        do_sample=True,
        top_k=50,
        top_p=1.0,
        temperature=0.9,
    ):
        self.device = device
        device_index = torch.device(device).index
        device_index = device_index if device_index is not None else torch.cuda.current_device()
        self.device_index = device_index

        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.num_layers = pred_config.num_hidden_layers
        self.hidden_size = pred_config.hidden_size
        self.num_code_groups = pred_config.num_code_groups
        self.num_codebooks = self.num_code_groups - 1  # 15
        self.max_seq = 2 + self.num_codebooks  # 17
        self.do_sample = do_sample
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature

        # Extract model components (references, not copies)
        cp = code_predictor
        self.small_to_mtp = cp.small_to_mtp_projection
        self.pred_model = cp.model
        self.lm_heads = cp.lm_head  # ModuleList[15]
        self.codec_embeds = cp.model.codec_embedding  # ModuleList[15]
        self.has_sliding_layers = "sliding_attention" in getattr(
            self.pred_model.config, "layer_types", []
        )

        B = max_batch_size

        # StaticCache with batch_size=B
        self.static_cache = StaticCache(
            config=pred_config, max_cache_len=self.max_seq, batch_size=B
        )

        # Cache position tensors (shared across batch — all samples at same positions)
        self.prefill_cache_pos = torch.arange(2, device=device)
        self.decode_cache_positions = [
            torch.tensor([2 + i], device=device) for i in range(self.num_codebooks - 1)
        ]

        # Static I/O buffers — [B, ...]
        self.input_buf = torch.zeros(B, 2, talker_hidden_size, dtype=dtype, device=device)
        self.output_tokens = torch.zeros(B, self.num_codebooks, dtype=torch.long, device=device)

        self.graph = None
        self.captured = False
        self.prefill_attn = None
        self.decode_attn = None

    def _init_cache_layers(self):
        """Force lazy initialization of StaticCache layers before graph capture."""
        config = self.pred_model.config
        num_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
        head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        B = self.max_batch_size
        dummy_k = torch.zeros(B, num_kv_heads, 1, head_dim, dtype=self.dtype, device=self.device)
        for layer in self.static_cache.layers:
            if not layer.is_initialized:
                layer.lazy_initialization(dummy_k)

    def _make_attn_mask(self, input_embeds: torch.Tensor, cache_position: torch.Tensor):
        mask = create_causal_mask(
            config=self.pred_model.config,
            input_embeds=input_embeds,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=self.static_cache,
        )
        if self.has_sliding_layers:
            sliding = create_sliding_window_causal_mask(
                config=self.pred_model.config,
                input_embeds=input_embeds,
                attention_mask=None,
                cache_position=cache_position,
                past_key_values=self.static_cache,
            )
            return {"full_attention": mask, "sliding_attention": sliding}
        return {"full_attention": mask}

    def _build_attention_masks(self):
        B = self.max_batch_size
        dummy_prefill = torch.zeros(B, 2, self.hidden_size, dtype=self.dtype, device=self.device)
        dummy_decode = torch.zeros(B, 1, self.hidden_size, dtype=self.dtype, device=self.device)
        self.prefill_attn = self._make_attn_mask(dummy_prefill, self.prefill_cache_pos)
        self.decode_attn = []
        for pos in self.decode_cache_positions:
            self.decode_attn.append(self._make_attn_mask(dummy_decode, pos))

    def _full_loop(self):
        """The full 15-step predictor loop on static buffers, batched."""
        B = self.max_batch_size

        # Project input from talker hidden size to predictor hidden size
        h = self.small_to_mtp(self.input_buf)  # [B, 2, pred_hidden]

        # Prefill: 2 tokens through all layers
        out = self.pred_model(
            inputs_embeds=h,
            attention_mask=self.prefill_attn,
            past_key_values=self.static_cache,
            cache_position=self.prefill_cache_pos,
            use_cache=True,
        )
        h = out.last_hidden_state  # [B, 2, pred_hidden]

        # First codebook: logits from last position
        logits = self.lm_heads[0](h[:, -1:, :])  # [B, 1, vocab]
        tok = batched_sample_logits(
            logits[:, 0, :],  # [B, vocab]
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            do_sample=self.do_sample,
        )  # [B]
        self.output_tokens[:, 0] = tok

        # Remaining 14 codebooks
        for cb_idx in range(1, self.num_codebooks):
            # Embed previous token using codebook-specific embedding
            emb = self.codec_embeds[cb_idx - 1](tok.unsqueeze(1))  # [B, 1, codec_hidden]
            emb = self.small_to_mtp(emb)  # [B, 1, pred_hidden]

            # Single-token decode through all layers
            out = self.pred_model(
                inputs_embeds=emb,
                attention_mask=self.decode_attn[cb_idx - 1],
                past_key_values=self.static_cache,
                cache_position=self.decode_cache_positions[cb_idx - 1],
                use_cache=True,
            )
            h = out.last_hidden_state  # [B, 1, pred_hidden]

            logits = self.lm_heads[cb_idx](h[:, -1:, :])  # [B, 1, vocab]
            tok = batched_sample_logits(
                logits[:, 0, :],
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                do_sample=self.do_sample,
            )
            self.output_tokens[:, cb_idx] = tok

        return self.output_tokens

    @torch.inference_mode()
    def capture(self, num_warmup=3):
        """Warmup and capture the CUDA graph."""
        print(f"Warming up batched predictor (B={self.max_batch_size}, {num_warmup} runs)...")

        self._init_cache_layers()
        self._build_attention_masks()

        for _ in range(num_warmup):
            self.static_cache.reset()
            self._full_loop()
        torch.cuda.synchronize()

        print("Capturing CUDA graph for batched predictor...")

        with torch.cuda.device(self.device_index):
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self.graph = torch.cuda.CUDAGraph()
                # Warmup in capture stream
                self.static_cache.reset()
                self._full_loop()
                torch.cuda.synchronize()

                self.static_cache.reset()
                with torch.cuda.graph(self.graph):
                    self._full_loop()

        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.captured = True
        print(f"Batched predictor CUDA graph captured! (B={self.max_batch_size})")

    @torch.inference_mode()
    def run(self, pred_input: torch.Tensor) -> torch.Tensor:
        """
        Run the captured graph.

        pred_input: [B, 2, talker_hidden_size]
            - B must equal max_batch_size (pad with zeros for unused slots)
        Returns: [B, 15] long tensor of codebook tokens
        """
        self.input_buf.copy_(pred_input)
        self.static_cache.reset()
        self.graph.replay()
        return self.output_tokens.clone()
