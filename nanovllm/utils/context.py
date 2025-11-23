from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    cu_seqlens_taskq: torch.Tensor | None = None    
    block_tables: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, cu_seq_lens_taskq=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, cu_seq_lens_taskq, block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()

from collections import defaultdict

class SimpleTimer: 
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.timings = defaultdict(float)
            cls._instance.counts = defaultdict(int)
        return cls._instance
    
    def add_time(self, func_name: str, elapsed_time: float):
        self.timings[func_name] += elapsed_time
        self.counts[func_name] += 1
        
    def reset(self):
        self.timings.clear()
        self.counts.clear()
    
    def print_stats(self):
        if not self.timings:
            return
        
        print("\n" + "="*60)
        print("计时统计")
        
        for name in sorted(self.timings.keys(), key=lambda x: self.timings[x], reverse=True):
            time_s = self.timings[name]
            count = self.counts[name]
            
            print(f"{name:<25} {time_s:>8.3f}s {count:>6}次")
        
        print("="*60)

timer = SimpleTimer()