#!/usr/bin/env python3
"""Aggregate lm-eval accuracy metrics with Fast-dLLM NFE sidecars.

Example:
    python scripts/aggregate_eval_nfe.py --results_root eval_results_nfe
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return data


def _find_nfe_stats(results_path: Path) -> dict[str, Any]:
    direct = results_path.parent / "nfe_stats.json"
    if direct.is_file():
        return _load_json(direct)
    ranked = sorted(results_path.parent.glob("nfe_stats*.json"))
    if ranked:
        return _load_json(ranked[0])
    return {}


def _variant_from_model_path(model_path: str, run_tag: str) -> str:
    src = f"{model_path} {run_tag}".lower()
    # Match in order of specificity: stop-grad before relay because the
    # relay-sg checkpoint dir is named ``bptt_relay_stopgrad_*``.
    if "bptt_relay_stopgrad" in src or "relay_sg" in src or "relay-sg" in src:
        return "relay-sg"
    if "bptt_relay" in src or src.endswith(" relay") or "/relay/" in src or " relay " in src:
        return "relay"
    if "vanilla" in src:
        return "vanilla"
    if "hf_baseline_fast_dllm_v2_1.5b" in src or "efficient-large-model" in src:
        return "hf-baseline"
    return "unknown"


def _checkpoint_from_model_path(model_path: str) -> str:
    m = re.search(r"checkpoint-(\d+)", model_path)
    if m:
        return f"checkpoint-{m.group(1)}"
    return "final"


def _is_metric_value(key: str, value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    low = key.lower()
    if "stderr" in low or low in {"alias"}:
        return False
    return True


def rows_for_results(results_path: Path) -> list[dict[str, Any]]:
    result = _load_json(results_path)
    nfe = _find_nfe_stats(results_path)
    run_tag = results_path.parents[1].name if len(results_path.parents) > 1 else ""
    model_path = str(nfe.get("model_path") or "")
    variant = _variant_from_model_path(model_path, run_tag)
    checkpoint = _checkpoint_from_model_path(model_path)
    rows: list[dict[str, Any]] = []

    task_results = result.get("results", {})
    if not isinstance(task_results, dict):
        return rows

    for task, metrics in sorted(task_results.items()):
        if not isinstance(metrics, dict):
            continue
        for metric, value in sorted(metrics.items()):
            if not _is_metric_value(metric, value):
                continue
            score = float(value)
            rows.append(
                {
                    "task": task,
                    "metric": metric,
                    "variant": variant,
                    "score": score,
                    "score_pct": score * 100.0,
                    "avg_nfe": nfe.get("avg_nfe", ""),
                    "avg_nfe_per_generated_token": nfe.get(
                        "avg_nfe_per_generated_token", ""
                    ),
                    "avg_generated_tokens": nfe.get("avg_generated_tokens", ""),
                    "median_nfe": nfe.get("median_nfe", ""),
                    "median_nfe_per_generated_token": nfe.get(
                        "median_nfe_per_generated_token", ""
                    ),
                    "median_generated_tokens": nfe.get("median_generated_tokens", ""),
                    "min_nfe": nfe.get("min_nfe", ""),
                    "max_nfe": nfe.get("max_nfe", ""),
                    "total_samples": nfe.get("total_samples", ""),
                    "checkpoint": checkpoint,
                    "threshold": nfe.get("threshold", ""),
                    "use_carry": nfe.get("use_carry", ""),
                    "small_block_size": nfe.get("small_block_size", ""),
                    "model_path": model_path,
                    "results_path": str(results_path),
                    "nfe_stats_path": str(results_path.parent / "nfe_stats.json")
                    if (results_path.parent / "nfe_stats.json").is_file()
                    else "",
                }
            )
    return rows


def latest_results_by_run_dir(root: Path) -> list[Path]:
    """Return the newest lm-eval results file from each model/run directory."""
    by_dir: dict[Path, list[Path]] = {}
    for path in root.rglob("results_*.json"):
        by_dir.setdefault(path.parent, []).append(path)
    return sorted(max(paths, key=lambda path: path.stat().st_mtime) for paths in by_dir.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results_root",
        type=Path,
        default=Path("eval_results_nfe"),
        help="Root to scan for lm-eval results_*.json files.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="TSV output path. Default: <results_root>/aggregate_accuracy_nfe.tsv",
    )
    args = ap.parse_args()

    root = args.results_root
    if not root.is_absolute():
        root = REPO_ROOT / root
    out_path = args.out or (root / "aggregate_accuracy_nfe.tsv")
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    rows: list[dict[str, Any]] = []
    for path in latest_results_by_run_dir(root):
        rows.extend(rows_for_results(path))

    fieldnames = [
        "task",
        "metric",
        "variant",
        "score",
        "score_pct",
        "avg_nfe",
        "avg_nfe_per_generated_token",
        "avg_generated_tokens",
        "median_nfe",
        "median_nfe_per_generated_token",
        "median_generated_tokens",
        "min_nfe",
        "max_nfe",
        "total_samples",
        "checkpoint",
        "threshold",
        "use_carry",
        "small_block_size",
        "model_path",
        "results_path",
        "nfe_stats_path",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
