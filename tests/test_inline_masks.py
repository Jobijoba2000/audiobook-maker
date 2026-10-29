"""Exact equivalence of captured causal/padding masks and rotary positions."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest,torch
from transformers.masking_utils import create_causal_mask
from epub_audio._vendor.faster_qwen3_tts_batch.adaptive_talker_graph import AdaptiveTalkerGraph
from epub_audio._vendor.faster_qwen3_tts_batch.batched_talker_graph import BatchedTalkerGraph

class InlineMaskTests(unittest.TestCase):
    def test_masks_and_positions_match_transformers_for_every_capacity(self):
        for dtype in (torch.float32,torch.bfloat16):
            for capacity in (768,1024,1536,3072):
                for position in (0,7,capacity//2,capacity-1):
                    graph=AdaptiveTalkerGraph.__new__(AdaptiveTalkerGraph)
                    graph.max_seq_len=3072
                    graph._key_positions=torch.arange(3072).reshape(1,-1)
                    graph.cache_position=torch.tensor([position])
                    graph.input_buf=torch.zeros(3,1,2,dtype=dtype)
                    graph._attention_mask_2d=torch.ones(3,capacity,dtype=torch.long)
                    graph._attention_mask_2d[1,:11]=0;graph._attention_mask_2d[2]=0
                    graph.attn_mask=torch.empty(3,1,1,capacity,dtype=dtype)
                    graph._mask_zero=torch.zeros((),dtype=dtype)
                    graph._mask_min=torch.full((),torch.finfo(dtype).min,dtype=dtype)
                    graph.rope_deltas=torch.tensor([[0.],[-11.],[3.]])
                    graph.position_ids=torch.zeros(3,3,1)
                    config=SimpleNamespace(_attn_implementation='eager')
                    cache=SimpleNamespace(is_compileable=True,is_sliding=[False],get_mask_sizes=lambda p,i:(capacity,0))
                    expected=create_causal_mask(config,graph.input_buf,graph._attention_mask_2d,graph.cache_position,cache)
                    with self.subTest(dtype=dtype,capacity=capacity,position=position),patch.object(BatchedTalkerGraph,'_decode_step') as forward:
                        graph._decode_step()
                        forward.assert_called_once_with()
                        torch.testing.assert_close(graph.attn_mask,expected,rtol=0,atol=0)
                        expected_positions=(graph.rope_deltas+position).unsqueeze(0).expand(3,-1,-1)
                        torch.testing.assert_close(graph.position_ids,expected_positions,rtol=0,atol=0)

if __name__=='__main__':unittest.main()
