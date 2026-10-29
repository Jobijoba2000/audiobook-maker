"""The additional CUDA graph must preserve codec sums and bfloat16 rounding."""
from types import SimpleNamespace
import unittest

import torch

from epub_audio._vendor.faster_qwen3_tts_batch.codec_embedding_graph import CodecEmbeddingGraph


class CodecEmbeddingTests(unittest.TestCase):
    def check_equivalence(self, device, dtype, capture):
        generator = torch.Generator().manual_seed(17)
        embeddings = [torch.nn.Embedding(64, 32, device=device, dtype=dtype) for _ in range(15)]
        talker = SimpleNamespace(config=SimpleNamespace(num_code_groups=16, hidden_size=32),
                                 code_predictor=SimpleNamespace(get_input_embeddings=lambda: embeddings))
        graph = CodecEmbeddingGraph(talker, 6, device, dtype)
        if capture:
            rng_before = torch.cuda.get_rng_state().clone()
            graph.capture()
            self.assertTrue(torch.equal(torch.cuda.get_rng_state(), rng_before))
        output_address = graph.output.data_ptr()
        for _ in range(4):
            codes = torch.randint(0, 64, (6, 15), generator=generator).to(device)
            hidden = torch.randn(6, 1, 32, generator=generator).to(device=device, dtype=dtype)
            # Expanded, non-contiguous padding is also used by real generation.
            text = torch.randn(1, 1, 32, generator=generator).to(device=device, dtype=dtype).expand(6, -1, -1)
            hiddens = [hidden] + [embed(codes[:, i].unsqueeze(1)) for i, embed in enumerate(embeddings)]
            expected = torch.cat(hiddens, dim=1).sum(1, keepdim=True) + text
            if capture:
                actual = graph.run(hidden, codes, text)
            else:
                graph.last_hidden.copy_(hidden)
                graph.codes.copy_(codes)
                graph.text_addend.copy_(text)
                graph._compute()
                actual = graph.output
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(actual.data_ptr(), output_address)

    def test_exact_cpu_results_with_changing_codes_and_broadcast_text(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype), torch.inference_mode():
                self.check_equivalence('cpu', dtype, False)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_replays_are_exact_and_do_not_consume_sampling_randomness(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype), torch.inference_mode():
                self.check_equivalence('cuda', dtype, True)


if __name__ == '__main__':
    unittest.main()
