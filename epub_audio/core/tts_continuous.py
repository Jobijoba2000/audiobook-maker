"""Continuous slot reuse with the existing model, reference and CUDA graphs."""
from collections import deque
import torch
import numpy as np
from epub_audio._vendor.faster_qwen3_tts_batch.batched_sampling import batched_sample_logits


class ContinuousBackend:
    def __init__(self,backend):
        self.backend=backend
        self.settings=backend.settings
        self.batch_size=backend.batch_size
        self.trace=[]

    def generate(self,texts,prompt,*,on_output=None,should_stop=None):
        backend=self.backend
        base=backend.base
        prepared=[base._prepare_generation(text=t,language='French',voice_clone_prompt=prompt,
            xvec_only=False,non_streaming_mode=True,append_silence=False) for t in texts]
        model,talker,config,*_=prepared[0]
        if len(texts)<=self.batch_size:
            initial=list(range(len(texts)))
        else:
            # Start the largest contexts so replacement prefills fit the current
            # shared cache position; tag routing restores original reading order.
            initial=sorted(range(len(texts)),key=lambda i:prepared[i][3].shape[1],reverse=True)[:self.batch_size]
        pending=deque(i for i in range(len(texts)) if i not in initial)
        suppress=torch.zeros(config.vocab_size,dtype=torch.bool,device=base.device)
        suppress[max(0,config.vocab_size-1024):]=True
        suppress[config.codec_eos_token_id]=False
        from epub_audio._vendor.faster_qwen3_tts_batch import continuous_generate as continuous
        def replacement():
            if not pending:
                return None
            index=pending.popleft()
            inputs=prepared[index]
            out=talker.forward(inputs_embeds=inputs[3],attention_mask=inputs[4],
                use_cache=True,output_hidden_states=True,return_dict=True,
                trailing_text_hidden=None,tts_pad_embed=inputs[6],generation_step=None,
                past_hidden=None,past_key_values=None)
            first=batched_sample_logits(out.logits[:,-1,:],temperature=self.settings['temperature'],
                top_k=50,top_p=1.,do_sample=self.settings['sampling'],suppress_mask=suppress,
                suppress_eos_mask=torch.ones(1,dtype=torch.bool,device=base.device),
                eos_id=config.codec_eos_token_id)[0]
            rd=getattr(talker,'rope_deltas',None)
            return continuous.SlotReplacement(past_key_values=out.past_key_values,prefill_len=inputs[3].shape[1],
                first_token=first,past_hidden=out.past_hidden,trailing_text_hiddens=inputs[5],
                text_len=inputs[5].shape[1],rope_delta=0. if rd is None else float(rd[0,0].item()),tag=index)
        def decode(index,codes,timing):
            if codes is None or timing.get('ended_with_eos') is not True:
                return None,base.sample_rate,timing
            ref=prepared[index][7]
            combined=torch.cat([ref.to(codes.device),codes],dim=0) if ref is not None else codes
            arrays,rate=model.speech_tokenizer.decode({'audio_codes':combined.unsqueeze(0)})
            audio=arrays[0]
            if hasattr(audio,'cpu'):
                audio=audio.flatten().cpu().numpy()
            audio=np.asarray(audio,dtype=np.float32).flatten()
            if ref is not None:
                audio=audio[int(ref.shape[0]/combined.shape[0]*len(audio)):]
            return audio,rate,timing
        outputs={}
        def done(index,codes,timing):
            self.trace.append(dict(tag=index,steps=timing['steps'],finished=timing['ended_with_eos'],
                                   replacements=timing.get('total_replaced')))
            if index in outputs:
                raise RuntimeError('Un passage continu a été renvoyé deux fois.')
            outputs[index]=decode(index,codes,timing)
            if on_output is not None:
                on_output(index, *outputs[index])
        remaining = deque(initial + list(pending))
        pending.clear()
        while remaining:
            candidates = sorted(remaining, key=lambda i: prepared[i][3].shape[1], reverse=True)
            initial = candidates[:self.batch_size]
            pending.extend(i for i in remaining if i not in initial)
            remaining.clear()
            chosen = [prepared[i] for i in initial]
            continuous.continuous_batched_generate(talker=talker,
                talker_input_embeds_list=[p[3] for p in chosen],attention_mask_list=[p[4] for p in chosen],
                trailing_text_hiddens_list=[p[5] for p in chosen],request_tags=initial,
                tts_pad_embed=chosen[0][6],config=config,predictor_graph=backend.predictor,
                talker_graph=backend.talker,try_get_replacement=replacement,on_slot_done=done,
                minimum_cache_position=max(p[3].shape[1] for p in prepared),should_stop=should_stop,
                max_new_tokens=self.settings['max_new_tokens'],temperature=self.settings['temperature'],
                top_k=50,top_p=1.,do_sample=self.settings['sampling'],
                repetition_penalty=self.settings['repetition_penalty'])
            remaining.extend(pending)
            pending.clear()
        if set(outputs) != set(range(len(texts))):
            raise RuntimeError('La synthèse continue n’a pas renvoyé tous les passages.')
        return [outputs[i] for i in range(len(texts))]
