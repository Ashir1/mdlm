#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Generate HumanEval+ / MBPP+ samples as JSONL for EvalPlus scoring (no vLLM / HF backend).

Prompting matches EvalPlus ``codegen.py`` (instruction + fenced problem in the user turn,
assistant prefix opens a ```python block; see ``evalplus.provider.utility.make_raw_chat_prompt``).

Run from ``Fast-dLLM/v2`` with the same Python env you use for ``eval.py`` (evalplus must be
installed only for dataset + sanitize utilities).

Examples::

    cd /path/to/Fast-dLLM/v2
    python scripts/generate_evalplus_jsonl.py \\
        --model_path Efficient-Large-Model/Fast_dLLM_v2_1.5B \\
        --dataset humaneval \\
        --output_jsonl evalplus_results/fastdllm_humaneval.jsonl

    python scripts/generate_evalplus_jsonl.py \\
        --model_path output_models/hf_baseline_Fast_dLLM_v2_1.5B \\
        --dataset mbpp \\
        --output_jsonl evalplus_results/fastdllm_mbpp.jsonl

    # Trainer ``checkpoint-N`` (no local configuration.py) — custom code is taken from
    # the base Hub model and weights from the checkpoint (see --code_model_path).

Score with EvalPlus (separate env with ``evalplus`` is fine; only the evaluate step runs here)::

    evalplus.evaluate --dataset humaneval --samples evalplus_results/fastdllm_humaneval.jsonl
    evalplus.evaluate --dataset mbpp --samples evalplus_results/fastdllm_mbpp.jsonl

Optional: ``--base_only`` on evaluate for HumanEval/MBPP *base* tests only.

Optional: record **absolute** per-problem NFE (mean denoising forward passes per example, same
definition as ``eval.py`` / Table 2 caption) into a sidecar JSON::

    python scripts/generate_evalplus_jsonl.py ... \\
        --output_jsonl evalplus_results/relay_humaneval.jsonl \\
        --nfe_stats_json evalplus_results/relay_humaneval_nfe.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import generation_functions  # noqa: E402
try:
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    from evalplus.provider.utility import make_raw_chat_prompt
    from evalplus.sanitize import sanitize
except ImportError as e:
    raise SystemExit(
        "evalplus is required for datasets + sanitize. From fast-dllm-v2/v2 run: "
        "pip install -e '.[eval]'\n"
        f"Original error: {e}"
    ) from e


# Same strings as evalplus/codegen.py (default instruct recipe)
INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following problem "
    "in a markdown code block:"
)
RESPONSE_PREFIX = (
    "Below is a Python script with a self-contained function that solves the problem "
    "and passes corresponding tests:"
)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def natural_task_order(task_ids: List[str]) -> List[str]:
    def key(t: str) -> int:
        return int(t.split("/")[-1])

    return sorted(task_ids, key=key)


def build_chat_prompts(
    problems: Dict[str, dict],
    tokenizer,
) -> Dict[str, str]:
    out = {}
    for task_id, task in problems.items():
        raw = task["prompt"].strip()
        # MBPP+ "prompt" is the task text; HumanEval+ uses the standard stub + docstring
        p = make_raw_chat_prompt(
            raw,
            INSTRUCTION_PREFIX,
            RESPONSE_PREFIX,
            tokenizer,
        )
        out[task_id] = p
    return out


def _dir_has_trust_remote_code_modules(model_dir: Path) -> bool:
    return (model_dir / "configuration.py").is_file() and (model_dir / "modeling.py").is_file()


# Keep these out of a checkpoint's config.json when merging onto a fresh AutoConfig
# (avoids clobbering class registration / metadata).
_CHECKPOINT_CONFIG_JSON_SKIP: frozenset[str] = frozenset(
    {
        "architectures",
        "model_type",
        "auto_map",
        "transformers_version",
        "torch_dtype",
    }
)


def _apply_checkpoint_config_json(config: object, checkpoint_dir: Path) -> None:
    """
    For Trainer ``checkpoint-*`` dirs that only have ``config.json`` (no
    ``configuration.py``), we still must build the same architecture as
    training (RELAY relay LayerNorm) before ``load_state_dict``; otherwise
    relay weights become ``unexpected_keys`` and ``use_carry`` is wrong.
    """
    p = checkpoint_dir / "config.json"
    if not p.is_file():
        return
    with p.open(encoding="utf-8") as f:
        raw: Dict[str, object] = json.load(f)
    n_set = 0
    for k, v in raw.items():
        if k in _CHECKPOINT_CONFIG_JSON_SKIP:
            continue
        try:
            setattr(config, k, v)
            n_set += 1
        except (AttributeError, TypeError):
            pass
    if n_set:
        print(
            f"[generate_evalplus_jsonl] applied {n_set} config fields from {p} "
            f"(use_relay={getattr(config, 'use_relay', None)}, "
            f"relay_layer={getattr(config, 'relay_layer', None)})"
        )


def _load_state_dict_from_checkpoint_dir(model_dir: Path) -> Dict[str, torch.Tensor]:
    """Load full state dict from a HF Trainer/DeepSpeed merge dir (safetensors or bin)."""
    single = model_dir / "model.safetensors"
    if single.is_file():
        from safetensors.torch import load_file

        return load_file(str(single))

    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        from safetensors.torch import load_file

        with open(index, encoding="utf-8") as f:
            idx = json.load(f)
        state: Dict[str, torch.Tensor] = {}
        weight_map: Dict[str, str] = idx.get("weight_map", {})
        by_shard: Dict[str, List[str]] = {}
        for key, filename in weight_map.items():
            by_shard.setdefault(filename, []).append(key)
        for filename, keys in by_shard.items():
            shard_path = model_dir / filename
            if not shard_path.is_file():
                raise OSError(f"Missing shard {shard_path} referenced in model.safetensors.index.json")
            part = load_file(str(shard_path))
            for k in keys:
                state[k] = part[k]
        return state

    pytorch_bin = model_dir / "pytorch_model.bin"
    if pytorch_bin.is_file():
        try:
            ckpt = torch.load(
                str(pytorch_bin), map_location="cpu", weights_only=True
            )
        except TypeError:
            ckpt = torch.load(str(pytorch_bin), map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            return ckpt["state_dict"]  # type: ignore[return-value]
        if isinstance(ckpt, dict):
            return ckpt  # type: ignore[return-value]
        raise OSError(f"Unexpected pytorch_model.bin content in {model_dir}")

    raise OSError(
        f"No model weights found in {model_dir} "
        "(expected model.safetensors, model.safetensors.index.json, or pytorch_model.bin). "
        "If you only have a DeepSpeed sharded global_step* dir, run zero_to_fp32.py merge first."
    )


def _attach_mdm(model: torch.nn.Module) -> None:
    model.mdm_sample = types.MethodType(  # type: ignore[attr-defined]
        generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample,
        model,
    )


def _build_from_in_tree_source(checkpoint_dir: Path) -> torch.nn.Module:
    """
    Build a Fast-dLLM v2 model **directly from the in-tree source**
    (``src/lmflow/models/fast_dllm/{configuration,modeling}.py``) using the
    checkpoint's own ``config.json`` so the RELAY config fields
    (``use_relay``, ``relay_layer``) are honored at construction time. This
    avoids ``trust_remote_code`` round-trips through (potentially stale)
    Hugging Face Hub caches whose ``modeling.py`` may predate the relay
    implementation, which would otherwise leave ``relay_layer_norm.*``
    weights as ``unexpected_keys`` at load time.

    Use the same Python the trainer used.
    """
    src_dir = REPO_ROOT / "src"
    fast_dllm_dir = src_dir / "lmflow" / "models" / "fast_dllm"
    if not _dir_has_trust_remote_code_modules(fast_dllm_dir):
        raise OSError(
            f"In-tree Fast-dLLM source not found at {fast_dllm_dir}. "
            "Pass --code_model_path to point at a Hub id or local dir with "
            "configuration.py + modeling.py + config.json instead."
        )
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from lmflow.models.fast_dllm.configuration import (  # noqa: E402
        Fast_dLLM_QwenConfig,
    )
    from lmflow.models.fast_dllm.modeling import (  # noqa: E402
        Fast_dLLM_QwenForCausalLM,
    )

    cfg_path = checkpoint_dir / "config.json"
    if not cfg_path.is_file():
        raise OSError(f"{checkpoint_dir} is missing config.json — cannot infer architecture.")
    with cfg_path.open(encoding="utf-8") as f:
        raw: Dict[str, object] = json.load(f)
    init_kwargs = {k: v for k, v in raw.items() if k not in _CHECKPOINT_CONFIG_JSON_SKIP}
    cfg = Fast_dLLM_QwenConfig(**init_kwargs)
    print(
        f"[generate_evalplus_jsonl] built Fast_dLLM_QwenConfig from {cfg_path} "
        f"(use_relay={cfg.use_relay}, relay_layer={cfg.relay_layer})"
    )
    model = Fast_dLLM_QwenForCausalLM(cfg)
    return model.to(dtype=torch.bfloat16)


def _load_state_into_model(
    model: torch.nn.Module, mp: Path
) -> torch.nn.Module:
    """Load checkpoint ``state_dict`` into ``model`` and report mismatches."""
    state = _load_state_dict_from_checkpoint_dir(mp)
    r = model.load_state_dict(state, strict=False)
    if r is not None:
        if getattr(r, "missing_keys", None):
            print(
                f"[generate_evalplus_jsonl] partial load — missing keys (often "
                f"lm_head when tied): {list(r.missing_keys)[:12]}"
            )
        if getattr(r, "unexpected_keys", None):
            print(
                f"[generate_evalplus_jsonl] unexpected keys in checkpoint: "
                f"{list(r.unexpected_keys)[:12]}"
            )
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    return model


def load_model(
    model_path: str,
    device: str,
    trust_remote_code: bool,
    code_model_path: Optional[str] = None,
) -> Tuple[torch.nn.Module, object]:
    """
    Load Fast-dLLM v2. HuggingFace Trainer ``checkpoint-*`` dirs typically
    contain ``config.json`` + weights but no ``configuration.py`` / ``modeling.py``,
    so ``from_pretrained`` cannot resolve ``auto_map`` locally.

    Default path: build the architecture directly from ``src/lmflow/models/fast_dllm``
    using the checkpoint's own ``config.json``, then load the checkpoint weights
    via ``load_state_dict``. This guarantees the in-tree relay module
    (``relay_layer_norm``) is present **before** the state dict is loaded.

    Override path: pass ``code_model_path`` to a Hub id or local tree with
    ``configuration.py`` + ``modeling.py`` + ``config.json`` to use that source
    via ``trust_remote_code`` instead.
    """
    mp = Path(model_path)
    if not mp.exists():
        # Hub id: single from_pretrained (assumes Hub has both code and weights)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.bfloat16,
        )
        model.eval()
        model = model.to(device)
        _attach_mdm(model)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        return model, tokenizer

    if _dir_has_trust_remote_code_modules(mp):
        # Self-contained local dir: configuration.py + modeling.py + weights.
        model = AutoModelForCausalLM.from_pretrained(
            str(mp),
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.bfloat16,
        )
    elif code_model_path is not None:
        # User-supplied code source for trust_remote_code: build with the
        # checkpoint's hyperparameters (so carry modules exist) then load
        # checkpoint weights. Use ``from_config`` so we don't pull weights
        # from ``code_model_path``.
        if not (mp / "config.json").is_file():
            raise OSError(
                f"{mp} is missing config.json. Cannot infer architecture."
            )
        cfg = AutoConfig.from_pretrained(
            code_model_path, trust_remote_code=trust_remote_code
        )
        _apply_checkpoint_config_json(cfg, mp)
        print(
            f"[generate_evalplus_jsonl] {mp} has no configuration.py/modeling.py; "
            f"loading model class from --code_model_path={code_model_path}"
        )
        model = AutoModelForCausalLM.from_config(
            cfg, trust_remote_code=trust_remote_code
        ).to(dtype=torch.bfloat16)
        model = _load_state_into_model(model, mp)
    else:
        print(
            f"[generate_evalplus_jsonl] {mp} has no configuration.py/modeling.py; "
            f"building from in-tree src/lmflow/models/fast_dllm + checkpoint config.json"
        )
        model = _build_from_in_tree_source(mp)
        model = _load_state_into_model(model, mp)

    model.eval()
    model = model.to(device)
    _attach_mdm(model)
    tokenizer = AutoTokenizer.from_pretrained(
        str(mp), trust_remote_code=trust_remote_code
    )
    return model, tokenizer


@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    device: torch.device,
    seq_lens: List[int],
    mask_id: int,
    max_new_tokens: int,
    bd_size: int,
    small_block_size: int,
    threshold: float,
    use_carry: bool,
    use_block_cache: bool,
    return_nfe_stats: bool = False,
) -> Tuple[List[str], Optional[List[int]]]:
    batched_input_ids = []
    max_len = 0
    min_len = 10**9
    for p in prompts:
        model_inputs = tokenizer([p], return_tensors="pt").to(device)
        ids = model_inputs["input_ids"]
        batched_input_ids.append(ids)
        max_len = max(max_len, ids.shape[1])
        min_len = min(min_len, ids.shape[1])

    padded = [
        torch.cat(
            [
                x,
                torch.full(
                    (1, max_len - x.shape[1]),
                    mask_id,
                    dtype=torch.long,
                    device=device,
                ),
            ],
            dim=1,
        )
        for x in batched_input_ids
    ]
    batched = torch.cat(padded, dim=0)
    sl = torch.tensor(seq_lens, device=device)

    raw = model.mdm_sample(
        batched,
        tokenizer=tokenizer,
        block_size=bd_size,
        small_block_size=small_block_size,
        max_new_tokens=max_new_tokens,
        mask_id=mask_id,
        min_len=min_len,
        seq_len=sl,
        use_block_cache=use_block_cache,
        threshold=threshold,
        use_carry=use_carry,
        return_nfe_stats=return_nfe_stats,
    )
    if return_nfe_stats:
        finished_samples, nfe_pack = raw
        nfes = [int(x) for x in nfe_pack["nfe"]]
    else:
        finished_samples = raw
        nfes = None

    texts = []
    for i, s in enumerate(seq_lens):
        row = finished_samples[i]
        texts.append(
            tokenizer.decode(
                row[s:],
                skip_special_tokens=True,
            )
        )
    return texts, nfes


def postprocess_raw(raw: str) -> str:
    t = raw.strip()
    if "```" in t:
        t = t.split("```")[0].rstrip()
    return t


def to_evalplus_solution(raw: str, task: dict) -> str:
    entry = task["entry_point"]
    raw = postprocess_raw(raw)
    sol = sanitize(raw, entrypoint=entry)
    if not sol.strip():
        sol = sanitize(raw, entrypoint=None)
    # If the model only echoed partial text, fall back to stub + generation
    if f"def {entry}" not in sol:
        stub = task["prompt"].rstrip()
        sol = sanitize(stub + "\n" + raw, entrypoint=entry)
    if not sol.strip() or f"def {entry}" not in sol:
        sol = (task["prompt"].rstrip() + "\n" + raw.strip()).strip()
    return sol


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_path", required=True, help="HF id or local checkpoint path")
    ap.add_argument(
        "--code_model_path",
        default=None,
        help=(
            "Override: if set, build the architecture via trust_remote_code from this "
            "Hub id or local tree (which must have configuration.py + modeling.py + config.json). "
            "Default (unset): instantiate Fast_dLLM_QwenForCausalLM directly from "
            "the in-tree src/lmflow/models/fast_dllm using the checkpoint's own config.json."
        ),
    )
    ap.add_argument(
        "--dataset",
        choices=("humaneval", "mbpp"),
        required=True,
    )
    ap.add_argument("--output_jsonl", required=True, help="Path to write EvalPlus JSONL")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--mask_id", type=int, default=151665)
    ap.add_argument("--bd_size", type=int, default=32)
    ap.add_argument("--small_block_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--use_carry",
        action="store_true",
        help="Enable 2-step relay-state carry inference (matches eval.py).",
    )
    ap.add_argument(
        "--use_block_cache",
        action="store_true",
        help=(
            "Enable Fast-dLLM v2 intra-block KV cache. Validated for vanilla "
            "SFT checkpoints only; relay (use_carry=True) runs should leave "
            "it off."
        ),
    )
    ap.add_argument(
        "--nfe_stats_json",
        default="",
        help=(
            "If set, write a sidecar JSON with absolute per-task NFE counts "
            "(active denoising forwards per problem; same counting as eval.py "
            "batch_sample). Omits length-normalized metrics."
        ),
    )
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.dataset == "humaneval":
        problems = get_human_eval_plus()
    else:
        problems = get_mbpp_plus()

    model, tokenizer = load_model(
        args.model_path,
        str(device),
        trust_remote_code=True,
        code_model_path=args.code_model_path,
    )
    use_carry = args.use_carry
    if use_carry:
        has_carry = bool(getattr(model.config, "use_relay", False))
        if not has_carry:
            print(
                "[warn] --use_carry set but config has use_relay=False; "
                "running without relay effect."
            )
            use_carry = False
    use_block_cache = args.use_block_cache
    if use_block_cache and use_carry:
        print(
            "[warn] --use_block_cache with --use_carry is not yet carry-correct; "
            "disabling block cache for this run."
        )
        use_block_cache = False

    chat_by_id = build_chat_prompts(problems, tokenizer)
    ordered_ids = natural_task_order(list(problems.keys()))

    solutions: Dict[str, str] = {}
    nfe_by_task: Dict[str, int] = {}
    record_nfe = bool(str(args.nfe_stats_json).strip())
    batch: List[str] = []
    batch_ids: List[str] = []
    batch_lens: List[int] = []

    def flush() -> None:
        nonlocal batch, batch_ids, batch_lens
        if not batch:
            return
        texts, nfes = generate_batch(
            model,
            tokenizer,
            batch,
            device,
            batch_lens,
            args.mask_id,
            args.max_new_tokens,
            args.bd_size,
            args.small_block_size,
            args.threshold,
            use_carry,
            use_block_cache,
            return_nfe_stats=record_nfe,
        )
        if nfes is not None:
            if len(nfes) != len(batch_ids):
                raise RuntimeError(
                    f"NFE list length {len(nfes)} != batch size {len(batch_ids)}"
                )
            for tid, n in zip(batch_ids, nfes):
                nfe_by_task[tid] = int(n)
        for tid, raw in zip(batch_ids, texts):
            solutions[tid] = to_evalplus_solution(raw, problems[tid])
        batch, batch_ids, batch_lens = [], [], []

    # Longest prompts last → less padding waste within batch (same trick as eval.py)
    id_by_len = sorted(
        ordered_ids,
        key=lambda t: len(tokenizer.encode(chat_by_id[t])),
    )

    for tid in tqdm(id_by_len, desc="generate"):
        prompt = chat_by_id[tid]
        enc = tokenizer([prompt], return_tensors="pt")
        batch.append(prompt)
        batch_ids.append(tid)
        batch_lens.append(enc["input_ids"].shape[1])
        if len(batch) >= args.batch_size:
            flush()
    flush()

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for tid in ordered_ids:
            rec = {"task_id": tid, "solution": solutions[tid]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(ordered_ids)} rows to {out_path}")

    if record_nfe:
        missing = [tid for tid in ordered_ids if tid not in nfe_by_task]
        if missing:
            raise RuntimeError(
                f"Missing NFE for {len(missing)} tasks (e.g. {missing[:5]})"
            )
        per_sample = [nfe_by_task[tid] for tid in ordered_ids]
        arr = np.asarray(per_sample, dtype=np.float64)
        nfe_path = Path(str(args.nfe_stats_json).strip())
        nfe_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_path": args.model_path,
            "dataset": args.dataset,
            "threshold": args.threshold,
            "bd_size": args.bd_size,
            "small_block_size": args.small_block_size,
            "max_new_tokens": args.max_new_tokens,
            "use_carry": use_carry,
            "use_block_cache": use_block_cache,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "mask_id": args.mask_id,
            "nfe_definition": (
                "Per-example active denoising forward calls in batch_sample; excludes "
                "prompt prefill and final cache-update next-token forwards."
            ),
            "task_ids": ordered_ids,
            "per_sample_nfe": per_sample,
            "total_samples": len(per_sample),
            "avg_nfe": float(arr.mean()),
            "median_nfe": float(np.median(arr)),
            "min_nfe": int(arr.min()),
            "max_nfe": int(arr.max()),
        }
        with nfe_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"Wrote NFE stats to {nfe_path}")


if __name__ == "__main__":
    main()
