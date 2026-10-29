"""EOS, inactive rows and repetition penalties remain correct without row syncs."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from epub_audio._vendor.faster_qwen3_tts_batch.batched_generate import batched_fast_generate
from epub_audio._vendor.faster_qwen3_tts_batch.batched_sampling import (
    batched_apply_repetition_penalty, batched_apply_repetition_penalty_mask, batched_sample_logits)


class FakeTalker:
    def __init__(self, actual, maximum):
        self.actual=actual
        self.maximum=maximum
        self.step=0
        self.embedding=torch.nn.Embedding(1030,2)
        self.code_predictor=SimpleNamespace(get_input_embeddings=lambda:[self.embedding]*15)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self,**kwargs):
        logits=torch.full((self.actual,1,1030),-100.0)
        for row in range(self.actual):
            logits[row,0,row+1]=100
        return SimpleNamespace(generation_step=0,logits=logits,
                               past_hidden=torch.zeros(self.actual,1,2),past_key_values=None)

    def codec_head(self,hidden):
        logits=torch.full((self.maximum,1030),-100.0)
        # Row zero ends immediately, row one takes three steps. Finished rows
        # subsequently emit non-EOS ids, but must never become active again.
        ids=([5,3] if self.step==0 else [1,4] if self.step==1 else [1,5])
        self.step+=1
        for row in range(self.maximum):
            logits[row,ids[row] if row<len(ids) else 1]=100
        return logits


class SyncOptimizationTests(unittest.TestCase):
    def test_penalty_matches_expected_values_for_flags_and_tensor_masks(self):
        source=torch.tensor([[2.,-4.,0.,8.],[3.,-6.,1.,9.],[-8.,4.,2.,0.]])
        histories=[torch.tensor([0,1,1]),torch.tensor([0,1]),torch.tensor([],dtype=torch.long)]
        expected=torch.tensor([[1.,-8.,0.,8.],[3.,-6.,1.,9.],[-8.,4.,2.,0.]])
        for mask in ([True,False,True],torch.tensor([True,False,True])):
            result=batched_apply_repetition_penalty(source.clone(),histories,2.,mask)
            torch.testing.assert_close(result,expected,rtol=0,atol=0)
        unchanged=source.clone()
        self.assertIs(batched_apply_repetition_penalty(unchanged,histories,1.,[True]*3),unchanged)
        torch.testing.assert_close(unchanged,source,rtol=0,atol=0)

    def test_gpu_presence_mask_matches_distinct_token_history_exactly(self):
        generator=torch.Generator().manual_seed(71)
        for dtype in (torch.float32,torch.bfloat16):
            for count in (2,3,4,5,6,7,8):
                source=torch.randn(count,1030,generator=generator).to(dtype)
                source[:,0]=0
                histories=[torch.randint(0,1030,(row*9,),generator=generator) for row in range(count)]
                active=torch.tensor([row%3!=1 for row in range(count)])
                seen=torch.zeros_like(source,dtype=torch.bool)
                for row,history in enumerate(histories):
                    seen[row,history]=True
                expected=batched_apply_repetition_penalty(source.clone(),histories,1.05,active)
                with patch.object(torch.Tensor,'__bool__',side_effect=AssertionError('Scalar mask check')):
                    result=batched_apply_repetition_penalty_mask(source,seen,1.05,active)
                torch.testing.assert_close(result,expected,rtol=0,atol=0)
                self.assertEqual(result.dtype,dtype)
                self.assertIs(batched_apply_repetition_penalty_mask(source,seen,1.,active),source)

    def test_global_and_per_row_eos_suppression_select_correct_tokens(self):
        source=torch.zeros(3,1030)
        source[:,2]=5
        source[:,5]=10
        source[:,1029]=100
        suppress=torch.zeros(1030,dtype=torch.bool)
        suppress[6:]=True
        with patch.object(torch.Tensor,'__bool__',side_effect=AssertionError('Scalar mask check')):
            tokens=batched_sample_logits(source,temperature=0.9,top_k=50,top_p=1.,do_sample=False,
                suppress_mask=suppress,suppress_eos_mask=torch.tensor([True,False,True]),eos_id=5)
        self.assertEqual(tokens.tolist(),[2,5,2])
        self.assertEqual(source[0,1029].item(),100)

    def generate(self,maximum,limit,row_limits=None,events=None,interval=1,cache_limit=128):
        talker=FakeTalker(2,maximum)
        graph=SimpleNamespace(max_batch_size=maximum,max_seq_len=cache_limit,
                              prefill_kv_batched=lambda *args:2,
                              set_generation_state=lambda *args,**kwargs:None,
                              run=lambda *args,**kwargs:torch.zeros(maximum,1,2))
        predictor=SimpleNamespace(run=lambda inputs:torch.zeros(maximum,15,dtype=torch.long))
        # A Python truth check on a CUDA tensor would require a scalar read.
        with patch('torch.cuda.synchronize'), patch.object(torch.Tensor,'__bool__',
                side_effect=AssertionError('Unexpected per-row tensor truth check')):
            return batched_fast_generate(talker=talker,
                talker_input_embeds_list=[torch.zeros(1,2,2)]*2,
                attention_mask_list=[torch.ones(1,2,dtype=torch.long)]*2,
                trailing_text_hiddens_list=[torch.zeros(1,1,2),torch.zeros(1,3,2)],
                tts_pad_embed=torch.zeros(1,1,2),
                config=SimpleNamespace(codec_eos_token_id=5,num_code_groups=16,vocab_size=1030),
                predictor_graph=predictor,talker_graph=graph,max_new_tokens=limit,
                min_new_tokens=0,do_sample=False,repetition_penalty=1.05,
                max_steps_per_sample=row_limits,
                completion_check_interval=interval,
                on_generated=(lambda row: events.append((row,talker.step))) if events is not None else None)

    def test_different_endings_and_unused_slots_keep_exact_code_order(self):
        for maximum in (2,3,4,5,6,7,8):
            with self.subTest(maximum=maximum):
                results=self.generate(maximum,5)
                self.assertEqual(len(results),2)
                self.assertEqual(results[0][0][:,0].tolist(),[1])
                self.assertEqual(results[1][0][:,0].tolist(),[2,3,4])
                self.assertTrue(all(timing['ended_with_eos'] for _,timing in results))
                self.assertEqual([timing['steps'] for _,timing in results],[1,3])

    def test_token_limit_still_marks_unfinished_passages_for_retry(self):
        results=self.generate(6,2)
        self.assertTrue(results[0][1]['ended_with_eos'])
        self.assertFalse(results[1][1]['ended_with_eos'])
        self.assertEqual(results[1][0][:,0].tolist(),[2,3])


    def test_row_budgets_preserve_eos_at_boundary_and_other_rows(self):
        cases = (([1,1],[1,1],[True,False],[False,True]),
                 ([1,2],[1,2],[True,False],[False,True]),
                 ([1,3],[1,3],[True,True],[False,False]),
                 ([4,4],[1,3],[True,True],[False,False]))
        for maximum in (2,3,4,5,6,7,8):
            for limits,steps,eos,exhausted in cases:
                with self.subTest(maximum=maximum,limits=limits):
                    results=self.generate(maximum,8,limits)
                    self.assertEqual([t['steps'] for _,t in results],steps)
                    self.assertEqual([t['ended_with_eos'] for _,t in results],eos)
                    self.assertEqual([t['estimated_limit_reached'] for _,t in results],exhausted)
                    self.assertEqual(results[0][0][:,0].tolist(),[1])
                    self.assertEqual(results[1][0][:,0].tolist(),[2,3,4][:steps[1]])

    def test_eos_events_are_individual_ignore_padding_and_never_report_timeouts(self):
        for maximum in (2,3,4,5,6,7,8):
            for limits,expected in ((None,[(0,1),(1,3)]),
                                    ([1,1],[(0,1)]),([1,2],[(0,1)]),
                                    ([1,3],[(0,1),(1,3)])):
                with self.subTest(maximum=maximum,limits=limits):
                    events=[]
                    results=self.generate(maximum,8,limits,events)
                    self.assertEqual(events,expected)
                    self.assertEqual([row for row,_ in events],
                                     [row for row,(_,t) in enumerate(results) if t['ended_with_eos']])
                    for row,(_,timing) in enumerate(results):
                        self.assertEqual(timing['estimated_limit_reached'],
                                         limits is not None and not timing['ended_with_eos'])

    def test_invalid_row_budgets_fail_before_generation(self):
        for limits in ([1],[1,2,3],[0,2],[-1,2],[9,2],[True,2],[1.5,2]):
            with self.subTest(limits=limits),self.assertRaises(ValueError):
                self.generate(6,8,limits)

    def test_global_budget_remains_the_fallback_without_row_limits(self):
        results=self.generate(6,2)
        self.assertEqual([t['generation_limit'] for _,t in results],[2,2])
        self.assertFalse(any(t['estimated_limit_reached'] for _,t in results))

    def test_periodic_host_polls_preserve_codes_limits_eos_and_final_notifications(self):
        for maximum in (2,6,8):
            for limit in (1,2,3,5,8):
                for limits in (None,[1,1],[1,2],[1,3]):
                    if limits is not None and max(limits)>limit:
                        continue
                    expected=self.generate(maximum,limit,limits)
                    for interval in (4,8):
                        with self.subTest(maximum=maximum,limit=limit,limits=limits,interval=interval):
                            events=[]
                            actual=self.generate(maximum,limit,limits,events,interval)
                            for (codes,timing),(reference,original) in zip(actual,expected):
                                if reference is None:
                                    self.assertIsNone(codes)
                                else:
                                    torch.testing.assert_close(codes,reference,rtol=0,atol=0)
                                for key in ('steps','ended_with_eos','estimated_limit_reached','generation_limit'):
                                    self.assertEqual(timing[key],original[key])
                            self.assertEqual(sorted(row for row,_ in events),
                                             [row for row,(codes,t) in enumerate(expected)
                                              if codes is not None and t['ended_with_eos']])

    def test_invalid_poll_intervals_are_rejected(self):
        for interval in (0,-1,9,True,1.5):
            with self.subTest(interval=interval),self.assertRaises(ValueError):
                self.generate(6,8,interval=interval)

    def test_cache_boundary_flushes_eos_between_polls_without_completing_partial_rows(self):
        events=[]
        results=self.generate(6,8,events=events,interval=4,cache_limit=5)
        self.assertEqual([row for row,_ in events],[0])
        self.assertEqual([t['steps'] for _,t in results],[1,3])
        self.assertEqual([t['ended_with_eos'] for _,t in results],[True,False])
        self.assertEqual(results[1][0][:,0].tolist(),[2,3,4])


if __name__=='__main__':
    unittest.main()
