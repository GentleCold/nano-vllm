import atexit
import copy
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int] | dict, sampling_params: SamplingParams):
        if isinstance(prompt, dict):
            text = prompt.get("text")
            task_start = prompt.get("task_start", None)
        else:
            text = prompt
            task_start = None

        if isinstance(text, str):
            token_ids = self.tokenizer.encode(text)
        elif isinstance(text, list):
            token_ids = copy(text)
        else:
            raise TypeError(f"Unsupported prompt type: {type(text)}")

        seq = Sequence(token_ids, sampling_params, task_start=task_start)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str | dict],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:    
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        
        outputs = {}
        step_count = 0
        start_time = perf_counter()
        
        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            schedule=schedule(
                wait=1,      # 跳过前1步
                warmup=1,    # 预热1步（不记录）
                active=5,    # 记录5步
                repeat=1
            ),
            on_trace_ready=tensorboard_trace_handler('./logs'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
        ) as prof:
            
            try:
                while not self.is_finished():
                    step_count += 1
                    
                    with record_function(f"## step_{step_count} ##"):
                        step_start = perf_counter()
                        
                        with record_function("model_step"):
                            output, num_tokens = self.step()
                        
                        step_time = perf_counter() - step_start
                        
                        # 更新进度条
                        if use_tqdm:
                            if num_tokens > 0:
                                throughput = num_tokens / step_time
                                pbar.set_postfix({
                                    "Prefill": f"{int(throughput)}tok/s",
                                    "StepTime": f"{step_time*1000:.1f}ms"
                                })
                            else:
                                throughput = -num_tokens / step_time
                                pbar.set_postfix({
                                    "Decode": f"{int(throughput)}tok/s",
                                    "StepTime": f"{step_time*1000:.1f}ms"
                                })
                        
                        # 收集输出
                        for seq_id, token_ids in output:
                            outputs[seq_id] = token_ids
                            if use_tqdm:
                                pbar.update(1)
                    
                    # 通知profiler完成一步
                    prof.step()
                    
            except Exception as e:
                print(f"生成过程中出错: {e}")
                raise
        
        # 计算总时间
        end_time = perf_counter()
        total_time = end_time - start_time
        print("总时间", total_time)
        
        # 处理输出
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} 
                for token_ids in outputs]
        
        if use_tqdm:
            pbar.close()
        
        return outputs
