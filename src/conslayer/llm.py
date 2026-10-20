"""Thin local-LLM wrapper (HF transformers, CPU, greedy decoding => deterministic)."""
from __future__ import annotations

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalLLM:
    def __init__(self, name: str = "Qwen/Qwen2.5-1.5B-Instruct", cache_dir: str | None = None,
                 threads: int | None = None):
        if threads:
            torch.set_num_threads(threads)
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(name, cache_dir=cache_dir, dtype=torch.bfloat16)
        self.model.eval()
        self.calls = 0

    @torch.inference_mode()
    def chat(self, system: str, user: str, max_new_tokens: int = 48) -> tuple[str, float]:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        t0 = time.perf_counter()
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=self.tok.eos_token_id)
        dt = time.perf_counter() - t0
        self.calls += 1
        text = self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return text, dt
