from types import SimpleNamespace
from unittest.mock import patch,Mock
import unittest,torch
from epub_audio._vendor.faster_qwen3_tts_batch.adaptive_talker_graph import AdaptiveTalkerGraph
from epub_audio._vendor.faster_qwen3_tts_batch.batched_talker_graph import BatchedTalkerGraph
from epub_audio.core.tts_batch import FasterBatchBackend

class CacheTests(unittest.TestCase):
    def fake(self):
        graph=AdaptiveTalkerGraph.__new__(AdaptiveTalkerGraph)
        graph.max_seq_len=3072
        graph._current_capacity=3072
        graph.force_full_capacity=False
        graph.capacity_counts={n:0 for n in graph.capacities}
        graph._cache_storage=[(torch.zeros(2,1,3072,2),torch.zeros(2,1,3072,2)) for _ in range(2)]
        graph.static_cache=SimpleNamespace(layers=[SimpleNamespace(is_sliding=False) for _ in range(2)])
        graph._full_attention_mask=torch.ones(2,3072,dtype=torch.long)
        graph._variants={n:(SimpleNamespace(replay=Mock()),torch.zeros(2,1,1,n)) for n in graph.capacities}
        graph.input_buf=torch.zeros(2,1,2)
        graph.output_buf=torch.zeros(2,1,2)
        graph.cache_position=torch.zeros(1,dtype=torch.long)
        graph._bind_cache_views(3072)
        graph.graph,graph.attn_mask=graph._variants[3072]
        return graph

    def test_growing_views_preserve_all_prefixes_and_share_storage(self):
        graph=self.fake()
        graph._select(768)
        for layer in graph.static_cache.layers:
            layer.keys[:,:,767,:]=17
            layer.values[:,:,767,:]=31
        graph._select(1024)
        for layer in graph.static_cache.layers:
            self.assertTrue(torch.all(layer.keys[:,:,767,:]==17))
            self.assertTrue(torch.all(layer.values[:,:,767,:]==31))
            layer.keys[:,:,1023,:]=23
            layer.values[:,:,1023,:]=37
        graph._select(1536)
        for layer,(keys,values) in zip(graph.static_cache.layers,graph._cache_storage):
            self.assertEqual(layer.keys.data_ptr(),keys.data_ptr())
            self.assertEqual(layer.values.data_ptr(),values.data_ptr())
            self.assertTrue(torch.all(layer.keys[:,:,767,:]==17))
            self.assertTrue(torch.all(layer.values[:,:,767,:]==31))
            self.assertTrue(torch.all(layer.keys[:,:,1023,:]==23))
            self.assertTrue(torch.all(layer.values[:,:,1023,:]==37))
            layer.keys[:,:,1535,:]=43
        graph._select(3072)
        self.assertTrue(all(torch.all(layer.keys[:,:,1535,:]==43) for layer in graph.static_cache.layers))
        self.assertEqual(graph.max_seq_len,3072)

    def test_positions_select_views_without_lowering_the_generation_limit(self):
        graph=self.fake()
        inputs=torch.ones_like(graph.input_buf)
        for position,capacity in [(0,768),(767,768),(768,1024),(1023,1024),(1024,1536),(1535,1536),(1536,3072),(3071,3072)]:
            with self.subTest(position=position):
                before=graph._variants[capacity][0].replay.call_count
                self.assertIs(graph.run(inputs,position),graph.output_buf)
                self.assertEqual(graph._current_capacity,capacity)
                self.assertEqual(graph.max_seq_len,3072)
                self.assertEqual(graph.cache_position.item(),position)
                self.assertTrue(torch.equal(graph.input_buf,inputs))
                self.assertEqual(graph._variants[capacity][0].replay.call_count,before+1)

    def test_capture_prepares_four_views_over_the_full_shared_storage(self):
        graph=self.fake()
        graph._init_cache_layers=lambda:None
        storage=[layer.keys.data_ptr() for layer in graph.static_cache.layers]
        captured=[]
        def capture(instance,prefill_len,num_warmup):
            captured.append((instance.max_seq_len,prefill_len,num_warmup))
            instance.graph=object()
            instance.attn_mask=torch.zeros(2,1,1,instance.max_seq_len)
        with patch.object(BatchedTalkerGraph,'capture',autospec=True,side_effect=capture):
            graph.capture(prefill_len=512,num_warmup=3)
        self.assertEqual(captured,[(768,512,3),(1024,512,3),(1536,512,3),(3072,512,3)])
        self.assertEqual(tuple(graph._variants),(768,1024,1536,3072))
        self.assertEqual([keys.data_ptr() for keys,_ in graph._cache_storage],storage)
        self.assertEqual(graph.max_seq_len,3072)
        self.assertEqual(graph._current_capacity,3072)

    def test_every_prefill_restores_the_full_allocation(self):
        graph=self.fake();graph._select(768)
        with patch.object(BatchedTalkerGraph,'prefill_kv_batched',return_value=9) as original:
            self.assertEqual(graph.prefill_kv_batched('cache',9,2),9)
            original.assert_called_once_with('cache',9,2)
        self.assertEqual(graph._current_capacity,3072)
        self.assertTrue(all(layer.keys.shape[2]==3072 for layer in graph.static_cache.layers))

    def test_baseline_keeps_full_capacity_for_short_positions(self):
        graph=self.fake();graph.force_full_capacity=True
        graph.run(torch.zeros_like(graph.input_buf),3)
        self.assertEqual(graph._current_capacity,3072)

    def test_invalid_positions_are_rejected_before_graph_replay(self):
        graph=self.fake()
        for position in (-1,3072):
            with self.subTest(position=position),self.assertRaises(ValueError):
                graph.run(None,position)

    def test_reset_restores_full_capacity_before_clearing_the_cache(self):
        graph=self.fake();graph._select(768)
        with patch.object(BatchedTalkerGraph,'reset') as original:
            graph.reset()
        original.assert_called_once_with()
        self.assertEqual(graph._current_capacity,3072)

    def test_backend_enables_adaptive_cache_by_default_and_can_keep_the_fixed_graph(self):
        config=SimpleNamespace(hidden_size=2)
        talker=SimpleNamespace(model=object(),code_predictor=SimpleNamespace(model=SimpleNamespace(config=object())))
        base=SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(talker=talker,config=SimpleNamespace(talker_config=config))),device='cuda',dtype=torch.bfloat16)
        settings=dict(sampling=True,temperature=.9)
        for enabled in (True,False):
            with self.subTest(adaptive=enabled), \
                 patch('epub_audio._vendor.faster_qwen3_tts_batch.batched_predictor_graph.BatchedPredictorGraph'), \
                 patch('epub_audio._vendor.faster_qwen3_tts_batch.codec_embedding_graph.CodecEmbeddingGraph') as codec, \
                 patch('epub_audio._vendor.faster_qwen3_tts_batch.batched_talker_graph.BatchedTalkerGraph') as fixed, \
                 patch('epub_audio._vendor.faster_qwen3_tts_batch.adaptive_talker_graph.AdaptiveTalkerGraph') as adaptive:
                backend=(FasterBatchBackend(base,settings,batch_size=6) if enabled else
                         FasterBatchBackend(base,settings,batch_size=6,adaptive_cache=False))
                chosen=adaptive if enabled else fixed
                other=fixed if enabled else adaptive
                chosen.assert_called_once()
                other.assert_not_called()
                self.assertIs(backend.talker,chosen.return_value)
                self.assertIs(backend.adaptive_cache,enabled)
                self.assertEqual(chosen.call_args.kwargs['max_seq_len'],3072)
                chosen.return_value.capture.assert_called_once_with(prefill_len=512,num_warmup=3)
                if enabled:
                    codec.assert_called_once_with(talker,6,base.device,base.dtype)
                    codec.return_value.capture.assert_called_once_with()
                    self.assertIs(backend.codec_embeddings,codec.return_value)
                else:
                    codec.assert_not_called()
                    self.assertIsNone(backend.codec_embeddings)

if __name__=='__main__':unittest.main()
