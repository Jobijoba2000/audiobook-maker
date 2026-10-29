"""Capture the fixed-shape codec embedding work between predictor and talker."""
import torch


class CodecEmbeddingGraph:
    def __init__(self, talker, batch_size, device, dtype):
        self.embeddings = talker.code_predictor.get_input_embeddings()
        groups = talker.config.num_code_groups
        hidden = talker.config.hidden_size
        self.last_hidden = torch.zeros(batch_size, 1, hidden, device=device, dtype=dtype)
        self.codes = torch.zeros(batch_size, groups - 1, device=device, dtype=torch.long)
        self.text_addend = torch.zeros_like(self.last_hidden)
        self.output = torch.zeros_like(self.last_hidden)
        self.graph = None

    def _compute(self):
        # Keep exactly the original concatenation and reduction order, including
        # the bfloat16 rounding before adding the text context.
        hiddens = [self.last_hidden]
        for index, embedding in enumerate(self.embeddings):
            hiddens.append(embedding(self.codes[:, index].unsqueeze(1)))
        self.output.copy_(torch.cat(hiddens, dim=1).sum(1, keepdim=True) + self.text_addend)

    @torch.inference_mode()
    def capture(self):
        with torch.cuda.device(self.codes.device):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._compute()
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    self._compute()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()

    @torch.inference_mode()
    def run(self, last_hidden, codes, text_addend):
        self.last_hidden.copy_(last_hidden)
        self.codes.copy_(codes)
        self.text_addend.copy_(text_addend)
        self.graph.replay()
        return self.output
