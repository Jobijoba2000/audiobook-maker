"""Adaptive attention views sharing the full 3072-position KV allocation."""
import torch
from .batched_talker_graph import BatchedTalkerGraph

class AdaptiveTalkerGraph(BatchedTalkerGraph):
    capacities=(768,1024,1536,3072)

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.max_seq_len!=3072:
            raise ValueError('Le cache adaptatif nécessite une capacité maximale de 3072 positions.')
        if self.model.config.sliding_window is not None:
            raise ValueError('Les graphes adaptatifs nécessitent une attention causale complète.')
        self._key_positions=torch.arange(self.max_seq_len,device=self.device).reshape(1,-1)
        self._mask_zero=torch.zeros((),device=self.device,dtype=self.dtype)
        self._mask_min=torch.full((),torch.finfo(self.dtype).min,device=self.device,dtype=self.dtype)
        self.force_full_capacity=False
        self.capacity_counts={n:0 for n in self.capacities}
        self._variants={}
        self._current_capacity=3072

    def _decode_step(self):
        # Full causal attention, one query position, and fixed GPU operands:
        # capture mask and rotary updates without host scalar construction.
        capacity=self._attention_mask_2d.shape[-1]
        allowed=(self._key_positions[:,:capacity]<=self.cache_position[0]) & self._attention_mask_2d.bool()
        self.attn_mask.copy_(torch.where(allowed[:,None,None,:],self._mask_zero,self._mask_min))
        delta=self.rope_deltas+self.cache_position[0].to(self.rope_deltas.dtype)
        self.position_ids.copy_(delta.unsqueeze(0).expand(3,-1,-1))
        return super()._decode_step()

    def _bind_cache_views(self,capacity):
        # Views share the full allocation, so growth preserves all previous KV.
        for layer,(keys,values) in zip(self.static_cache.layers,self._cache_storage):
            if getattr(layer,'is_sliding',False):
                raise ValueError('Le cache adaptatif ne prend pas en charge les fenêtres glissantes.')
            layer.keys=keys[:,:,:capacity,:]
            layer.values=values[:,:,:capacity,:]
            layer.max_cache_len=capacity
        self._attention_mask_2d=self._full_attention_mask[:,:capacity]
        self._current_capacity=capacity

    def _select(self,capacity):
        if self._current_capacity==capacity:
            return
        self._bind_cache_views(capacity)
        self.graph,self.attn_mask=self._variants[capacity]

    @torch.inference_mode()
    def capture(self,prefill_len=512,num_warmup=3):
        self._init_cache_layers()
        self._cache_storage=[(layer.keys,layer.values) for layer in self.static_cache.layers]
        self._full_attention_mask=self._attention_mask_2d
        for capacity in self.capacities:
            self._bind_cache_views(capacity)
            self.max_seq_len=capacity
            super().capture(prefill_len=min(prefill_len,capacity-1),num_warmup=num_warmup)
            self._variants[capacity]=(self.graph,self.attn_mask)
        # The generator always sees the original upper bound, including when a
        # narrower graph is running. It can never stop at a narrow-view border.
        self.max_seq_len=3072
        self._current_capacity=3072

    def reset(self):
        self._select(3072)
        return super().reset()

    def prefill_kv_batched(self,*args,**kwargs):
        self._select(3072)
        # Reset the entire allocation before each batch, not just its prefix.
        return super().prefill_kv_batched(*args,**kwargs)

    def prefill_kv(self,*args,**kwargs):
        self._select(3072)
        return super().prefill_kv(*args,**kwargs)

    @torch.inference_mode()
    def run(self,input_embeds,position):
        if not 0 <= position < self.max_seq_len:
            raise ValueError('Position incompatible avec la capacité du cache adaptatif.')
        capacity=(3072 if self.force_full_capacity else
                  next(n for n in self.capacities if position<n))
        self._select(capacity)
        self.capacity_counts[capacity]+=1
        self.input_buf.copy_(input_embeds)
        self.cache_position[0]=position
        self.graph.replay()
        return self.output_buf
