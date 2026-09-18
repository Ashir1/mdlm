# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

'''
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
'''
try:
    import dotenv
except ImportError:
    dotenv = None

if dotenv is not None:
    dotenv.load_dotenv(override=True, verbose=True)  # set env variables from .env file

import accelerate
import torch
import re
from pathlib import Path
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import json
import time
import types
import generation_functions


def _model_arg_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).lower() in ("true", "1", "yes")


def _lm_base_device_is_readonly() -> bool:
    """Newer lm-eval: ``LM.device`` is a read-only ``@property`` (store on ``_device``)."""
    desc = getattr(LM, "device", None)
    return isinstance(desc, property) and getattr(desc, "fset", None) is None


def _set_eval_harness_device(harness: "Fast_dLLM_v2EvalHarness", dev: torch.device) -> None:
    """Compatible with lm-eval versions that use either ``_device`` or a plain ``device`` attr."""
    if _lm_base_device_is_readonly():
        harness._device = dev
    else:
        harness.device = dev


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("fast_dllm_v2")
class Fast_dLLM_v2EvalHarness(LM):
    def __init__(
        self,
        model_path='Efficient-Large-Model/Fast_dLLM_v2_7B',
        device="cuda",
        show_speed=False,
        max_new_tokens=2048,
        batch_size=32,
        mask_id=151665,
        use_block_cache=False,
        small_block_size=8,
        bd_size=32,
        threshold=0.9,
        use_carry=False,
        nfe_stats_path="",
        debug_print=False,
        debug_print_prompt=False,
        **kwargs,
    ):

        super().__init__()

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None
        
        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16, 
            **model_kwargs
        )
        self.model.eval()

        self.model.mdm_sample = types.MethodType(generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample, self.model)

        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            resolved = torch.device(str(self.accelerator.device))
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model = self.model.to(device)
            resolved = torch.device(device)
            self._rank = 0
            self._world_size = 1

        _set_eval_harness_device(self, resolved)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        self.show_speed = show_speed
        # ``model_args`` arrives as strings from lm_eval CLI, so coerce.
        self.max_new_tokens = int(max_new_tokens)
        self.batch_size = int(batch_size)
        self.mask_id = int(mask_id)
        self.model_path = model_path
        self.use_block_cache = _model_arg_bool(use_block_cache)
        self.small_block_size = int(small_block_size)
        self.threshold = float(threshold)
        self.bd_size = int(bd_size)
        self.use_carry = str(use_carry).lower() in ("true", "1", "yes")
        self.nfe_stats_path = str(nfe_stats_path or "").strip()
        self._nfe_values = []
        self._nfe_generated_token_counts = []
        self._nfe_task_names = set()
        self._has_carry = bool(getattr(self.model.config, "use_relay", False))
        if self.use_carry:
            if self._has_carry:
                print(
                    "[eval] use_carry=True -- will run 2-step forward with the "
                    f"RELAY relay state (use_relay=True, "
                    f"relay_layer={getattr(self.model.config, 'relay_layer', -1)})"
                )
            else:
                print(
                    "[eval] use_carry=True requested but model config has "
                    "use_relay=False; falling back to single-pass."
                )

        self._debug_print_prompt = _model_arg_bool(debug_print_prompt)
        _ = _model_arg_bool(debug_print)  # lm-eval may pass it; trajectory hook removed

    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size

    @property
    def tokenizer_name(self):
        return self.model_path
    
    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        return self.tokenizer.apply_chat_template(chat_history, add_generation_prompt=add_generation_prompt, tokenize=False)
    
    def loglikelihood_rolling(self, requests):
        raise NotImplementedError
    
    def _encode_pair(self, context, continuation):
        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc


    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        batch[:, prompt_index.sum()] = self.mask_id

        batch = torch.cat([batch.to(self.device), torch.full((b, self.bd_size-batch.shape[1]%self.bd_size), self.mask_id, dtype=torch.long, device=self.device)], dim=1)
        if batch.shape[1] > l:
            batch[:, l] = self.tokenizer.eos_token_id

        return batch

    @torch.no_grad()
    def get_logits(self, batch):
        out = self.model(batch)
        logits = out.logits
        logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)

        if self.use_carry and self._has_carry and out.h_s is not None:
            # Right-shift the carried hidden state by one position to mirror
            # the logit shift; the raw ``h_s[i]`` is the representation that
            # was used to predict token ``i+1``, so we shift to align with
            # position ``i`` before feeding it back as ``h_t``.
            h_s = out.h_s
            h_s = torch.cat([h_s[:, :1, :], h_s[:, :-1, :]], dim=1)
            out2 = self.model(batch, h_t=h_s)
            logits = out2.logits
            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)

        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []

        perturbed_seq = self._forward_process(seq.clone(), prompt_index)

        mask_indices = perturbed_seq == self.mask_id

        logits = self.get_logits(perturbed_seq)
        seq = torch.cat([seq.to(self.device), torch.full((seq.shape[0], self.bd_size-seq.shape[1]%self.bd_size), -100, dtype=torch.long, device=self.device)], dim=1)
        loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none')
        loss = loss.sum()
        loss_acc.append(loss.item())

        return - sum(loss_acc) / len(loss_acc)


    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)
                out.append((ll, 0.0))
        torch.cuda.empty_cache()
        return out

    def _write_nfe_stats(self):
        if not self.nfe_stats_path:
            return
        vals = np.asarray(self._nfe_values, dtype=np.float64)
        gen_lens = np.asarray(self._nfe_generated_token_counts, dtype=np.float64)
        if vals.size and gen_lens.size != vals.size:
            raise RuntimeError(
                f"NFE stats mismatch: {vals.size} NFE values but "
                f"{gen_lens.size} generated-token counts"
            )
        denom = np.maximum(gen_lens, 1.0) if gen_lens.size else gen_lens
        norm_vals = vals / denom if vals.size else vals
        payload = {
            "model_path": self.model_path,
            "tasks": sorted(self._nfe_task_names),
            "threshold": self.threshold,
            "bd_size": self.bd_size,
            "small_block_size": self.small_block_size,
            "max_new_tokens": self.max_new_tokens,
            "use_carry": self.use_carry,
            "use_block_cache": self.use_block_cache,
            "rank": self.rank,
            "world_size": self.world_size,
            "total_samples": int(vals.size),
            "nfe_definition": "Per-example active denoising forward calls in batch_sample; excludes prompt prefill and final cache-update next-token forwards.",
            "nfe_per_generated_token_definition": "per_sample_nfe divided by generated continuation token count after removing mask and pad tokens; denominator is clamped to at least 1.",
            "per_sample_nfe": [int(x) for x in self._nfe_values],
            "per_sample_generated_tokens": [
                int(x) for x in self._nfe_generated_token_counts
            ],
            "per_sample_nfe_per_generated_token": [
                float(x) for x in norm_vals.tolist()
            ],
        }
        if vals.size:
            payload.update(
                {
                    "avg_nfe": float(vals.mean()),
                    "median_nfe": float(np.median(vals)),
                    "min_nfe": int(vals.min()),
                    "max_nfe": int(vals.max()),
                    "avg_generated_tokens": float(gen_lens.mean()),
                    "median_generated_tokens": float(np.median(gen_lens)),
                    "avg_nfe_per_generated_token": float(norm_vals.mean()),
                    "median_nfe_per_generated_token": float(np.median(norm_vals)),
                    "min_nfe_per_generated_token": float(norm_vals.min()),
                    "max_nfe_per_generated_token": float(norm_vals.max()),
                }
            )
        else:
            payload.update(
                {
                    "avg_nfe": None,
                    "median_nfe": None,
                    "min_nfe": None,
                    "max_nfe": None,
                    "avg_generated_tokens": None,
                    "median_generated_tokens": None,
                    "avg_nfe_per_generated_token": None,
                    "median_nfe_per_generated_token": None,
                    "min_nfe_per_generated_token": None,
                    "max_nfe_per_generated_token": None,
                }
            )
        path = Path(self.nfe_stats_path)
        if self.world_size > 1:
            path = path.with_name(f"{path.stem}.rank{self.rank}{path.suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"[eval] Wrote NFE stats to {path}", flush=True)
    
    def generate_until(self, requests):
        output = [None] * len(requests)  # pre-allocate output list
        num_tokens = 0
        
        start_time = time.time()
        
        requests_with_indices = [(i, req) for i, req in enumerate(requests)]
        requests_with_indices.sort(key=lambda x: len(x[1].args[0]))
        
        batched_requests = []
        current_batch = []
        for i, req in requests_with_indices:
            current_batch.append((i, req))
            if len(current_batch) == self.batch_size:
                batched_requests.append(current_batch)
                current_batch = []
        
        if current_batch:
            batched_requests.append(current_batch)

        for _, batch in enumerate(tqdm(batched_requests, desc="Generating...")):
            batched_input_ids = []
            max_len = 0
            min_len = 1e9
            seq_len = []
            
            for orig_idx, req in batch:
                question = req.args[0]
                
                if req.task_name.startswith('minerva_math'):
                    question = question.replace("Solution:", "Please reason step by step, and put your final answer within \\boxed{{}}.")
                elif req.task_name.startswith('gsm8k'):
                    question = question.replace("Answer:", "Please reason step by step, and put your final answer within \\boxed{{}}.")
                model_inputs = self.tokenizer([question], return_tensors="pt").to(self.device)
                batched_input_ids.append(model_inputs["input_ids"])
                max_len = max(max_len, model_inputs["input_ids"].shape[1])
                min_len = min(min_len, model_inputs["input_ids"].shape[1])
                seq_len.append(model_inputs["input_ids"].shape[1])
            
            # pad batched_input_ids to the same length
            batched_input_ids = [torch.cat([input_ids, torch.full((1, max_len - input_ids.shape[1]), self.mask_id, dtype=torch.long, device=self.device)], dim=1) for input_ids in batched_input_ids]
            batched_input_ids = torch.cat(batched_input_ids, dim=0)
            batched_input_ids = batched_input_ids.to(self.device)
            
            with torch.no_grad():
                if self.accelerator is not None:
                    generated_ids = self.accelerator.unwrap_model(self.model).mdm_sample(
                        batched_input_ids,
                        tokenizer=self.tokenizer,
                        block_size=self.bd_size,
                        small_block_size=self.small_block_size,
                        max_new_tokens=self.max_new_tokens,
                        mask_id=self.mask_id,
                        min_len=min_len,
                        seq_len=torch.tensor(seq_len, device=self.device),
                        use_block_cache=self.use_block_cache,
                        threshold=self.threshold,
                        use_carry=self.use_carry,
                        return_nfe_stats=bool(self.nfe_stats_path),
                    )
                else:
                    generated_ids = self.model.mdm_sample(
                        batched_input_ids,
                        tokenizer=self.tokenizer,
                        block_size=self.bd_size,
                        small_block_size=self.small_block_size,
                        max_new_tokens=self.max_new_tokens,
                        mask_id=self.mask_id,
                        min_len=min_len,
                        seq_len=torch.tensor(seq_len, device=self.device),
                        use_block_cache=self.use_block_cache,
                        threshold=self.threshold,
                        use_carry=self.use_carry,
                        return_nfe_stats=bool(self.nfe_stats_path),
                    )
                nfe_stats = None
                if self.nfe_stats_path:
                    generated_ids, nfe_stats = generated_ids
                    self._nfe_values.extend(int(x) for x in nfe_stats.get("nfe", []))
                    self._nfe_task_names.update(req.task_name for _, req in batch)
            
            # extract new generated tokens, and keep original index order
            for batch_pos, (orig_idx, req) in enumerate(batch):
                generated_tail = generated_ids[batch_pos][seq_len[batch_pos]:]
                if self.nfe_stats_path:
                    valid_generated = generated_tail != self.mask_id
                    pad_id = self.tokenizer.pad_token_id
                    if pad_id is not None:
                        valid_generated = valid_generated & (generated_tail != pad_id)
                    self._nfe_generated_token_counts.append(
                        int(valid_generated.sum().item())
                    )
                generated_answer = self.tokenizer.decode(
                    generated_tail,
                    skip_special_tokens=True
                )
            
                # count token number
                if self.show_speed:
                    num_tokens += (generated_ids[batch_pos][seq_len[batch_pos]:] != self.mask_id).sum()
                
                # put result in the correct original index position
                output[orig_idx] = generated_answer

                if self._debug_print_prompt:
                    print('=' * 20)
                    print('question: ', req.args[0])
                    print('answer: ', generated_answer)
                    print('=' * 20, end='\n\n')
            
        end_time = time.time()
        if self.show_speed:
            print(f"Total number of tokens generated: {num_tokens}")
            print(f"Total time taken: {end_time - start_time} seconds")
            print(f"Tokens per second: {num_tokens / (end_time - start_time)}")
        self._write_nfe_stats()
            
        return output


if __name__ == "__main__":
    cli_evaluate()
    
