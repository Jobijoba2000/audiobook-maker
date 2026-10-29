"""Slot reuse, cache compaction, cancellation, output order and recovery."""
from collections import deque
from concurrent.futures import CancelledError
from pathlib import Path
from types import SimpleNamespace
import struct,tempfile,unittest,wave
from unittest.mock import Mock,patch
import torch
import numpy as np
from test_batch_sync import FakeTalker
from epub_audio._vendor.faster_qwen3_tts_batch.continuous_generate import continuous_batched_generate,SlotReplacement
from epub_audio._vendor.faster_qwen3_tts_batch.batched_talker_graph import BatchedTalkerGraph
from epub_audio.core.tts_generate import render_chapters
from epub_audio.core.tts_qwen import QwenEngine
from epub_audio.cli.args import arguments


class ContinuousTests(unittest.TestCase):
    def run_fake(self,count=2,limit=10,capacity=128,cancel=False):
        model=FakeTalker(2,2)
        if count>2:
            def head(hidden):
                logits=torch.full((2,1030),-100.)
                logits[:,5]=100
                return logits
            model.codec_head=head
        shifts=[]
        states=[]
        graph=SimpleNamespace(max_batch_size=2,max_seq_len=capacity,
            prefill_kv_batched=lambda *args:2,set_generation_state=lambda *args,**kwargs:None,
            replace_slot_kv=Mock(),update_slot_state=lambda *args:states.append(args),
            compact_cache=lambda position,shift:(shifts.append(shift) or position-shift),
            run=lambda *args,**kwargs:torch.zeros(2,1,2))
        pending=deque(range(2,count))
        def replace():
            if not pending:return None
            return SlotReplacement(past_key_values=None,prefill_len=2,first_token=torch.tensor(1),
                past_hidden=torch.zeros(1,1,2),trailing_text_hiddens=torch.zeros(1,1,2),
                text_len=1,rope_delta=0.,tag=pending.popleft())
        with patch('torch.cuda.synchronize'):
            results=continuous_batched_generate(talker=model,
                talker_input_embeds_list=[torch.zeros(1,2,2)]*2,
                attention_mask_list=[torch.ones(1,2,dtype=torch.long)]*2,
                trailing_text_hiddens_list=[torch.zeros(1,1,2)]*2,request_tags=[0,1],
                tts_pad_embed=torch.zeros(1,1,2),config=SimpleNamespace(
                    codec_eos_token_id=5,num_code_groups=16,vocab_size=1030),
                predictor_graph=SimpleNamespace(run=lambda x:torch.zeros(2,15,dtype=torch.long)),
                talker_graph=graph,try_get_replacement=replace,max_new_tokens=limit,
                minimum_cache_position=2,min_new_tokens=0,do_sample=False,
                should_stop=lambda:cancel)
        return results,shifts,states

    def test_final_single_passage_uses_individual_generation(self):
        engine = QwenEngine(batching=True, batch_size=6)
        engine.model = object()
        engine.prompt = object()
        engine.batch_backend = object()
        ready = Mock()
        path = Path('last.wav')
        with patch('epub_audio.core.tts_retry.synthesize_with_fallback', return_value=path) as generate:
            self.assertEqual(engine.synthesize_continuous(['Dernier passage.'], [path], on_ready=ready), [path])
        generate.assert_called_once_with(engine, 'Dernier passage.', path)
        ready.assert_called_once_with(0, path)

    def test_distinct_endings_and_token_limits_remain_per_passage(self):
        results,_,_=self.run_fake()
        values={tag:(codes[:,0].tolist(),timing['ended_with_eos']) for tag,codes,timing in results}
        self.assertEqual(values,{0:([1],True),1:([2,3,4],True)})
        results,_,_=self.run_fake(limit=2)
        values={tag:timing['ended_with_eos'] for tag,_,timing in results}
        self.assertEqual(values,{0:True,1:False})

    def test_many_replacements_survive_repeated_cache_compaction(self):
        results,shifts,states=self.run_fake(count=48,capacity=12)
        self.assertEqual(sorted(tag for tag,_,_ in results),list(range(48)))
        self.assertEqual(len(results),48)
        self.assertTrue(all(timing['ended_with_eos'] and timing['steps']==1 for _,_,timing in results))
        self.assertGreaterEqual(len(shifts),2)
        self.assertEqual(len(states),46)
        self.assertTrue(all(state[3]<=0 for state in states))

    def test_cancellation_stops_before_generating_further_tokens(self):
        with self.assertRaises(CancelledError):self.run_fake(cancel=True)

    def test_cache_compaction_preserves_keys_and_absolute_rotary_positions(self):
        graph=BatchedTalkerGraph.__new__(BatchedTalkerGraph)
        graph.max_seq_len=12
        keys=torch.arange(48).reshape(2,1,12,2).float()
        values=keys+100
        graph.static_cache=SimpleNamespace(layers=[SimpleNamespace(keys=keys.clone(),values=values.clone())])
        graph._attention_mask_2d=torch.ones(2,12,dtype=torch.long)
        graph._attention_mask_2d[:,:3]=0
        graph.rope_deltas=torch.tensor([[-3.],[-2.]])
        absolute=10+graph.rope_deltas.clone()
        position=graph.compact_cache(10,3)
        self.assertEqual(position,7)
        layer=graph.static_cache.layers[0]
        torch.testing.assert_close(layer.keys[:,:,:7],keys[:,:,3:10],rtol=0,atol=0)
        torch.testing.assert_close(layer.values[:,:,:7],values[:,:,3:10],rtol=0,atol=0)
        self.assertEqual(layer.keys[:,:,7:].count_nonzero().item(),0)
        torch.testing.assert_close(position+graph.rope_deltas,absolute,rtol=0,atol=0)

    def test_streaming_wavs_keep_source_order_pauses_and_progress_by_six(self):
        class Voice:
            serial_gpu=True
            continuous_batching=True
            batch_size=6
            max_segment_characters=None
            def synthesize_continuous(self,texts,paths,*,on_ready,should_stop):
                for index in reversed(range(len(texts))):
                    with wave.open(str(paths[index]),'wb') as audio:
                        audio.setparams((1,2,24000,0,'NONE','not compressed'))
                        audio.writeframes(struct.pack('<h',(index+1)*1000)*2400)
                    on_ready(index,paths[index])
                return paths
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            messages=[]
            tracks=list(render_chapters(Voice(),[['x'*210+str(i) for i in range(30)]],
                root,workers=1,progress=messages.append))
            self.assertEqual([m.done for m in messages],[6,12,18,24,30])
            with wave.open(str(tracks[0][1]),'rb') as audio:
                expected=(b'\0\0'*19200).join(struct.pack('<h',n*1000)*2400 for n in range(1,31))
                self.assertEqual(audio.readframes(audio.getnframes()),expected)

    def test_partial_continuous_failure_retries_only_unsaved_passages(self):
        engine=QwenEngine(batching=True)
        engine.model=object();engine.prompt=[object()];engine.batch_backend=object()
        engine._save_wave=Mock(side_effect=lambda text,path,*args,**kwargs:path)
        backend=Mock()
        def generate(texts,prompt,*,on_output,should_stop):
            on_output(0,np.ones(100,dtype=np.float32),24000,dict(ended_with_eos=True,steps=1,decode_s=1))
            raise RuntimeError('test interruption')
        backend.generate.side_effect=generate
        ready=[]
        with tempfile.TemporaryDirectory() as folder:
            paths=[Path(folder)/f'{i}.wav' for i in range(3)]
            with patch('epub_audio.core.tts_continuous.ContinuousBackend',return_value=backend), \
                 patch('epub_audio.core.tts_retry.synthesize_with_fallback',side_effect=lambda e,t,p,**kwargs:p) as retry, \
                 patch('torch.cuda.empty_cache'):
                result=engine.synthesize_continuous(['One','Two','Three'],paths,on_ready=lambda i,p:ready.append(i))
            self.assertEqual(result,paths)
            self.assertEqual([call.args[1] for call in retry.call_args_list],['Two','Three'])
            self.assertEqual(ready,[0,1,2])

    def test_cli_allows_return_to_length_groups(self):
        self.assertFalse(arguments(['--no-continuous-batching']).continuous_batching)
        self.assertTrue(arguments(['--continuous-batching']).continuous_batching)
        self.assertFalse(QwenEngine(batching=True).continuous_batching)
        self.assertTrue(QwenEngine(batching=True,continuous_batching=True).continuous_batching)
        with self.assertRaises(ValueError):QwenEngine(continuous_batching='yes')


if __name__=='__main__':unittest.main()
