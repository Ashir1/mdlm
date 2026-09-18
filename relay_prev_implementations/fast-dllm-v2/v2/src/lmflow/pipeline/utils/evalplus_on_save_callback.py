"""Trainer callback that submits EvalPlus jobs after checkpoint saves."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from transformers import TrainerCallback


class EvalPlusOnSaveCallback(TrainerCallback):
    def __init__(
        self,
        *,
        run_name: str,
        use_carry: bool,
        datasets: str = "humaneval mbpp",
        seeds: str = "0",
        threshold: float = 0.85,
        sbatch_path: str = "train_scripts/evalplus_checkpoint.sbatch",
        block_cache: str = "auto",
        strict: bool = False,
    ) -> None:
        self.run_name = run_name
        self.use_carry = use_carry
        self.datasets = datasets
        self.seeds = seeds
        self.threshold = threshold
        self.sbatch_path = sbatch_path
        self.block_cache = block_cache
        self.strict = strict

    @staticmethod
    def _is_world_process_zero(args: Any, state: Any) -> bool:
        if hasattr(state, "is_world_process_zero"):
            return bool(state.is_world_process_zero)
        return int(getattr(args, "process_index", 0)) == 0

    @staticmethod
    def _has_required_checkpoint_files(checkpoint_dir: Path) -> bool:
        if not (checkpoint_dir / "config.json").is_file():
            return False
        weight_files = (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
        return any((checkpoint_dir / name).is_file() for name in weight_files)

    def _pending_dir(self, args: Any) -> Path:
        return Path(args.output_dir) / "evalplus_results" / "on_save" / "pending"

    def _write_pending_manifest(
        self,
        *,
        args: Any,
        checkpoint_dir: Path,
        reason: str,
        env: dict[str, str],
    ) -> None:
        pending = self._pending_dir(args)
        pending.mkdir(parents=True, exist_ok=True)
        path = pending / f"{checkpoint_dir.name}.json"
        payload = {
            "checkpoint_dir": str(checkpoint_dir),
            "reason": reason,
            "created_at": time.time(),
            "sbatch_path": self.sbatch_path,
            "env": env,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if not self._is_world_process_zero(args, state):
            return control

        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        marker = checkpoint_dir / ".evalplus_submitted"
        if marker.is_file():
            return control

        if not self._has_required_checkpoint_files(checkpoint_dir):
            msg = f"checkpoint not ready for EvalPlus submission: {checkpoint_dir}"
            if self.strict:
                raise RuntimeError(msg)
            print(f"[EvalPlusOnSaveCallback] {msg}")
            return control

        sbatch_path = Path(self.sbatch_path)
        if not sbatch_path.is_absolute():
            sbatch_path = Path.cwd() / sbatch_path

        env = os.environ.copy()
        exported = {
            "MODEL_PATH": str(checkpoint_dir),
            "TRAIN_RUN_NAME": self.run_name,
            "CHECKPOINT_STEP": str(state.global_step),
            "EVALPLUS_DATASETS": self.datasets,
            "EVALPLUS_SEEDS": self.seeds,
            "EVALPLUS_THRESHOLD": str(self.threshold),
            "EVALPLUS_USE_CARRY": "1" if self.use_carry else "0",
            "EVALPLUS_BLOCK_CACHE": self.block_cache,
            "EVALPLUS_WANDB_GROUP": self.run_name,
        }
        env.update(exported)

        try:
            result = subprocess.run(
                ["sbatch", str(sbatch_path)],
                env=env,
                cwd=str(Path.cwd()),
                text=True,
                capture_output=True,
                check=True,
            )
        except FileNotFoundError as e:
            msg = "sbatch is not available; wrote pending EvalPlus manifest"
            self._write_pending_manifest(args=args, checkpoint_dir=checkpoint_dir, reason=str(e), env=exported)
            if self.strict:
                raise RuntimeError(msg) from e
            print(f"[EvalPlusOnSaveCallback] {msg}")
            return control
        except subprocess.CalledProcessError as e:
            reason = f"sbatch failed with exit code {e.returncode}: {e.stderr or e.stdout}"
            self._write_pending_manifest(args=args, checkpoint_dir=checkpoint_dir, reason=reason, env=exported)
            if self.strict:
                raise RuntimeError(reason) from e
            print(f"[EvalPlusOnSaveCallback] {reason}")
            return control

        submission = (result.stdout or result.stderr or "").strip()
        marker.write_text(
            json.dumps(
                {
                    "submitted_at": time.time(),
                    "submission": submission,
                    "sbatch_path": str(sbatch_path),
                    "env": exported,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"[EvalPlusOnSaveCallback] submitted EvalPlus job for {checkpoint_dir}: {submission}")
        return control
