from collections import deque
import threading

from termcolor import colored

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        # Concurrency: background prefetch thread and main thread both access scheduler
        self._lock = threading.Lock()
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.first = True
        self._recent_text_id: int | None = None

    def is_finished(self):
        with self._lock:
            return not self.waiting and not self.running

    def add(self, seq: Sequence):
        with self._lock:
            if self.waiting and seq.text_id is not None:
                waiting_list = list(self.waiting)
                insert_idx = None
                for i in range(len(waiting_list) - 1, -1, -1):
                    if waiting_list[i].text_id == seq.text_id:
                        insert_idx = i + 1
                        break
                if insert_idx is None and self._recent_text_id == seq.text_id:
                    insert_idx = 0
                if insert_idx is not None:
                    waiting_list.insert(insert_idx, seq)
                    self.waiting = deque(waiting_list)
                    return
            self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        with self._lock:
        # prefill
            if self.waiting and self._recent_text_id is not None:
                waiting_list = list(self.waiting)
                try:
                    idx = next(
                        i for i, seq in enumerate(waiting_list)
                        if seq.text_id == self._recent_text_id
                    )
                except StopIteration:
                    self._recent_text_id = None
                else:
                    if idx != 0:
                        seq = waiting_list.pop(idx)
                        waiting_list.insert(0, seq)
                        self.waiting = deque(waiting_list)
            scheduled_seqs = []
            num_seqs = 0
            num_batched_tokens = 0
            while self.waiting and num_seqs < self.max_num_seqs:
                seq = self.waiting[0]
                if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                    break
                num_seqs += 1
                self.block_manager.allocate(seq)
                num_batched_tokens += len(seq) - seq.num_cached_tokens
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
                scheduled_seqs.append(seq)
            if scheduled_seqs:
                for seq in scheduled_seqs:
                    if seq.text_id is not None:
                        self._recent_text_id = seq.text_id
                        break
                else:
                    self._recent_text_id = None
                return scheduled_seqs, True

            # decode
            # TODO: decode bug

            # while self.running and num_seqs < self.max_num_seqs:
            #     seq = self.running.popleft()
            #     while not self.block_manager.can_append(seq):
            #         if self.running:
            #             self.preempt(self.running.pop())
            #         else:
            #             self.preempt(seq)
            #             break
            #     else:
            #         num_seqs += 1
            #         self.block_manager.may_append(seq)
            #         scheduled_seqs.append(seq)

            assert scheduled_seqs
            self.running.extendleft(reversed(scheduled_seqs))
            return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        # called only under self._lock
        print(colored("preempt", "red"))
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        with self._lock:
            for seq, token_id in zip(seqs, token_ids):
                seq.append_token(token_id)
                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    if not seq.lock_block:
                        self.block_manager.deallocate(seq)
                    try:
                        self.running.remove(seq)
                    except Exception as e:
                        print(seq.status)
                        raise e

