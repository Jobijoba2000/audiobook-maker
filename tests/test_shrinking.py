"""CPU checks for shrinking history, row routing and memory safeguards."""
from collections import Counter
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from epub_audio.core.tts_batch import FasterBatchBackend
from epub_audio.core.tts_qwen import QwenEngine
from epub_audio.core.tts_shrinking import ShrinkingBatch, _Predictor, _Talker, _width_view


class ShrinkingTests(unittest.TestCase):
    def coordinator(self, size=6):
        owner = ShrinkingBatch.__new__(ShrinkingBatch)
        owner.maximum, owner.device = size, 'cpu'
        storage = [(torch.arange(size * 16).reshape(size, 1, 8, 2).clone(),
                    torch.arange(size * 16).reshape(size, 1, 8, 2).clone() + 1000)]
        talker = SimpleNamespace(_cache_storage=storage,
                                _full_attention_mask=torch.arange(size * 8).reshape(size, 8).clone(),
                                rope_deltas=torch.arange(size).reshape(size, 1).clone(),
                                max_seq_len=8, prefill_kv_batched=Mock(return_value=2),
                                set_generation_state=Mock())
        owner.backend = SimpleNamespace(talker=talker)
        owner.talkers = {n: SimpleNamespace(run=Mock(side_effect=lambda x, p: x + 100))
                         for n in range(1, size + 1)}
        owner.predictors = {n: SimpleNamespace(run=Mock(side_effect=lambda x: x + 10))
                            for n in range(1, size + 1)}
        owner.maps = {n: torch.arange(n) for n in range(1, size + 1)}
        owner.predictor_output = torch.zeros(size, 1, 2)
        owner.talker_output = torch.zeros(size, 1, 2)
        owner.predictor, owner.talker = _Predictor(owner), _Talker(owner)
        owner.reset([20] * size)
        return owner

    def test_repeated_compaction_preserves_history_masks_and_offsets(self):
        owner = self.coordinator()
        owner.prefill = 2
        talker = owner.backend.talker
        keys, values = talker._cache_storage[0]
        original_keys, original_values = keys.clone(), values.clone()
        original_mask, original_rope = talker._full_attention_mask.clone(), talker.rope_deltas.clone()
        previous_end = 0
        for finished, step, remaining in (([1, 4], 1, [0, 2, 3, 5]), ([0, 3], 2, [2, 5]), ([2], 3, [5])):
            end = owner.prefill + step
            # Simulate newly appended KV entries in the current compact layout.
            for position, original_row in enumerate(owner.layout):
                keys[position, :, previous_end:end].copy_(original_keys[original_row, :, previous_end:end])
                values[position, :, previous_end:end].copy_(original_values[original_row, :, previous_end:end])
            for row in finished:
                owner.on_generated(row)
            owner.select(step)
            self.assertEqual(owner.layout, remaining)
            torch.testing.assert_close(keys[:len(remaining), :, :end], original_keys[remaining, :, :end])
            torch.testing.assert_close(values[:len(remaining), :, :end], original_values[remaining, :, :end])
            torch.testing.assert_close(talker._full_attention_mask[:len(remaining)], original_mask[remaining])
            torch.testing.assert_close(talker.rope_deltas[:len(remaining)], original_rope[remaining])
            previous_end = end

    def test_outputs_return_to_original_rows_for_every_supported_size(self):
        for size in range(2, 9):
            with self.subTest(size=size):
                owner = self.coordinator(size)
                inputs = torch.arange(size * 2, dtype=torch.float32).reshape(size, 1, 2)
                remaining = list(range(1, size, 2))
                for row in set(range(size)) - set(remaining):
                    owner.on_generated(row)
                predicted = owner.predictor.run(inputs).clone()
                spoken = owner.talker.run(inputs, 3).clone()
                for row in range(size):
                    torch.testing.assert_close(predicted[row], inputs[row] + 10 if row in remaining else torch.zeros(1, 2))
                    torch.testing.assert_close(spoken[row], inputs[row] + 100 if row in remaining else torch.zeros(1, 2))
                self.assertEqual(owner.diagnostics()['batch_min_size'], len(remaining))

    def test_partial_batch_removes_padding_and_reset_restores_full_layout(self):
        owner = self.coordinator()
        owner.reset([10, 10, 10])
        self.assertEqual(owner.talker.prefill_kv_batched('inputs'), 2)
        owner.talker.set_generation_state('state')
        owner.backend.talker.set_generation_state.assert_called_once_with('state')
        owner.predictor.run(torch.ones(6, 1, 2))
        self.assertEqual(owner.layout, [0, 1, 2])
        owner.reset([10] * 6)
        self.assertEqual(owner.layout, list(range(6)))
        self.assertEqual(owner.finished, set())
        self.assertEqual(owner.steps_by_width, Counter())
        for limits in ([], [0], [True], [1] * 7):
            with self.assertRaises(ValueError):
                owner.reset(limits)

    def test_budget_expiry_does_not_count_as_successful_completion(self):
        owner = self.coordinator()
        owner.reset([1, 10, 10, 10, 10, 10])
        owner.select(1)
        self.assertEqual(owner.layout, [1, 2, 3, 4, 5])
        self.assertEqual(owner.finished, set())
        for row in range(1, 6):
            owner.on_generated(row)
        owner.select(10)
        self.assertEqual(owner.layout, [1, 2, 3, 4, 5])  # Ignored final host-poll steps.
        with self.assertRaises(ValueError):
            owner.on_generated(6)

    def test_sparse_widths_preserve_survivors_and_history(self):
        owner = self.coordinator()
        owner.prefill = 4
        owner.talkers = {n: owner.talkers[n] for n in (1, 2, 4, 6)}
        keys = owner.backend.talker._cache_storage[0][0]
        original = keys.clone()
        for row in (0, 2, 4):
            owner.on_generated(row)
        owner.select(0)
        self.assertEqual(owner.layout[:3], [1, 3, 5])
        self.assertEqual(len(owner.layout), 4)
        torch.testing.assert_close(keys[:3, :, :4], original[[1, 3, 5], :, :4])
        owner.on_generated(3)
        owner.select(0)
        self.assertEqual(owner.layout, [1, 5])
        torch.testing.assert_close(keys[:2, :, :4], original[[1, 5], :, :4])

    def test_partial_batch_can_use_larger_available_graph(self):
        owner = self.coordinator()
        owner.talkers = {n: owner.talkers[n] for n in (1, 2, 4, 6)}
        owner.reset([10, 10, 10])
        owner.select(0)
        self.assertEqual(owner.layout[:3], [0, 1, 2])
        self.assertEqual(len(owner.layout), 4)
        self.assertTrue(set(owner.layout[3:]).issubset(owner.finished))

    def test_graph_views_share_storage_and_pool_but_not_cache_metadata(self):
        owner = self.coordinator()
        parent = owner.backend.talker
        parent.graph = SimpleNamespace(pool=Mock(return_value=(1, 2)))
        parent.max_batch_size = 6
        parent.static_cache = SimpleNamespace(layers=[SimpleNamespace(keys=k, values=v)
                                                       for k, v in parent._cache_storage])
        parent.input_buf = torch.zeros(6, 1, 2)
        parent.output_buf = torch.zeros(6, 1, 2)
        parent.output_tokens = torch.zeros(6, 15, dtype=torch.long)
        parent.position_ids = torch.zeros(3, 6, 1)
        parent.capacities = (4, 8)
        for predictor in (False, True):
            with self.subTest(predictor=predictor):
                narrowed = _width_view(parent, 3, predictor=predictor)
                self.assertIsNot(narrowed.static_cache, parent.static_cache)
                self.assertIsNot(narrowed.static_cache.layers[0], parent.static_cache.layers[0])
                self.assertEqual(narrowed.static_cache.layers[0].keys.data_ptr(), parent._cache_storage[0][0].data_ptr())
                self.assertEqual(narrowed._capture_pool, (1, 2))
                self.assertEqual(narrowed.input_buf.shape[0], 3)
                self.assertEqual(parent.static_cache.layers[0].keys.shape[0], 6)

    def test_engine_mode_defaults_and_incompatible_modes(self):
        for batching, continuous, requested, expected in (
                (True, False, True, True), (True, False, False, False),
                (True, True, True, False), (False, False, True, False)):
            engine = QwenEngine(batching=batching, continuous_batching=continuous, shrinking_batching=requested)
            self.assertIs(engine.shrinking_batching, expected)
        with self.assertRaises(ValueError):
            FasterBatchBackend(None, {}, adaptive_cache=False, shrinking_batching=True)

    def test_graph_allocation_failure_keeps_fixed_backend(self):
        config = SimpleNamespace(hidden_size=2)
        talker = SimpleNamespace(code_predictor=SimpleNamespace(model=SimpleNamespace(config=object())), model=object())
        model = SimpleNamespace(talker=talker, config=SimpleNamespace(talker_config=config))
        base = SimpleNamespace(model=SimpleNamespace(model=model), device='cpu', dtype=torch.float32)
        settings = dict(sampling=True, temperature=0.9)
        with patch('epub_audio._vendor.faster_qwen3_tts_batch.batched_predictor_graph.BatchedPredictorGraph') as predictor, \
             patch('epub_audio._vendor.faster_qwen3_tts_batch.adaptive_talker_graph.AdaptiveTalkerGraph') as talker_graph, \
             patch('epub_audio._vendor.faster_qwen3_tts_batch.codec_embedding_graph.CodecEmbeddingGraph'), \
             patch('epub_audio.core.tts_shrinking.ShrinkingBatch', side_effect=torch.cuda.OutOfMemoryError('fixture')), \
             patch('torch.cuda.empty_cache') as release:
            backend = FasterBatchBackend(base, settings, shrinking_batching=True)
        self.assertIsNone(backend.shrinking)
        self.assertIn('lots fixes conservés', backend.shrinking_error)
        self.assertIs(backend.predictor, predictor.return_value)
        self.assertIs(backend.talker, talker_graph.return_value)
        release.assert_called_once()

    def test_cache_is_released_before_each_valid_wave_only(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                events = []
                backend = FasterBatchBackend.__new__(FasterBatchBackend)
                backend.batch_size = 3
                backend.settings = dict(max_new_tokens=2048, temperature=0.9, sampling=True, repetition_penalty=1.05)
                def decode(_):
                    events.append('decode')
                    return [np.zeros(40, dtype=np.float32)], 24000
                model = SimpleNamespace(speech_tokenizer=SimpleNamespace(decode=Mock(side_effect=decode)))
                backend.base = SimpleNamespace(sample_rate=24000, _prepare_generation=Mock(
                    return_value=(model, object(), object(), object(), object(), object(), object(), None)))
                backend.predictor, backend.talker, backend.codec_embeddings = object(), object(), None
                backend.shrinking = Mock() if enabled else None
                if enabled:
                    backend.shrinking.diagnostics.return_value = {'shrinking_batching': True}
                codes = torch.zeros(2, 16, dtype=torch.long)
                def generate(**kwargs):
                    if enabled:
                        kwargs['on_generated'](0)
                    return [(codes, dict(ended_with_eos=True)), (None, dict(ended_with_eos=False)),
                            (codes, dict(ended_with_eos=True))]
                with patch('epub_audio._vendor.faster_qwen3_tts_batch.batched_generate.batched_fast_generate',
                           side_effect=generate) as generated, \
                     patch('torch.cuda.empty_cache', side_effect=lambda: events.append('release')):
                    outputs = backend.generate(['One', 'Two', 'Three'], object())
                self.assertEqual(events, ['release', 'decode', 'release', 'decode'] if enabled else ['decode', 'decode'])
                self.assertIsNone(outputs[1][0])
                if enabled:
                    backend.shrinking.on_generated.assert_called_once_with(0)
                    self.assertIs(generated.call_args.kwargs['predictor_graph'], backend.shrinking.predictor)
