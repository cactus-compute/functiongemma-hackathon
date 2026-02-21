#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import optuna


MAIN_FILE = Path("main.py")
BENCH_CMD = ["./cactus/venv/bin/python", "benchmark.py"]
RESULT_RE = re.compile(r"TOTAL SCORE:\s*([0-9.]+)%")

PARAMS = [
    "FAIL_FAST_COMPLEXITY",
    "CONFIDENCE_BASE",
    "CONFIDENCE_SCALE",
    "INTENT_WEIGHT",
    "ARG_DIFFICULTY_WEIGHT",
    "TOOL_PRESSURE_WEIGHT",
    "TOOL_RELIABILITY_WEIGHT",
]

SEED_PARAMS = {
    "FAIL_FAST_COMPLEXITY": 0.38,
    "CONFIDENCE_BASE": 0.85,
    "CONFIDENCE_SCALE": 0.25,
    "INTENT_WEIGHT": 0.45,
    "ARG_DIFFICULTY_WEIGHT": 0.25,
    "TOOL_PRESSURE_WEIGHT": 0.10,
    "TOOL_RELIABILITY_WEIGHT": 0.25,
}


def patch_constants(text: str, params: dict) -> str:
    updated = text
    for name in PARAMS:
        value = params[name]
        pattern = rf"(^\s*{name}\s*=\s*)([0-9]*\.?[0-9]+)"
        updated, count = re.subn(
            pattern,
            rf"\g<1>{value:.4f}",
            updated,
            count=1,
            flags=re.MULTILINE,
        )
        if count == 0:
            raise RuntimeError(f"Could not find constant {name} in main.py")
    return updated


def run_benchmark(timeout_s: int) -> tuple[float, str]:
    proc = subprocess.run(
        BENCH_CMD,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(f"benchmark failed (exit {proc.returncode})\n{out}")
    m = RESULT_RE.search(out)
    if not m:
        raise RuntimeError(f"TOTAL SCORE not found in output\n{out}")
    return float(m.group(1)), out


def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "FAIL_FAST_COMPLEXITY": trial.suggest_float("FAIL_FAST_COMPLEXITY", 0.25, 0.55),
        "CONFIDENCE_BASE": trial.suggest_float("CONFIDENCE_BASE", 0.65, 0.95),
        "CONFIDENCE_SCALE": trial.suggest_float("CONFIDENCE_SCALE", 0.10, 0.45),
        "INTENT_WEIGHT": trial.suggest_float("INTENT_WEIGHT", 0.20, 0.60),
        "ARG_DIFFICULTY_WEIGHT": trial.suggest_float("ARG_DIFFICULTY_WEIGHT", 0.10, 0.60),
        "TOOL_PRESSURE_WEIGHT": trial.suggest_float("TOOL_PRESSURE_WEIGHT", 0.05, 0.30),
        "TOOL_RELIABILITY_WEIGHT": trial.suggest_float("TOOL_RELIABILITY_WEIGHT", 0.10, 0.45),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bayesian optimization for generate_hybrid constants")
    parser.add_argument("--trials", type=int, default=12, help="Number of Bayesian trials")
    parser.add_argument("--timeout", type=int, default=900, help="Per-trial benchmark timeout (seconds)")
    parser.add_argument("--results-file", default="bayes_sweep_results.jsonl", help="JSONL results output")
    args = parser.parse_args()

    original_text = MAIN_FILE.read_text()
    results_path = Path(args.results_file)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.enqueue_trial(SEED_PARAMS)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        patched = patch_constants(original_text, params)
        MAIN_FILE.write_text(patched)

        t0 = time.time()
        try:
            score, output = run_benchmark(timeout_s=args.timeout)
            elapsed = time.time() - t0
            record = {
                "trial": trial.number,
                "score": score,
                "elapsed_s": elapsed,
                "params": params,
            }
            with results_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(f"[trial {trial.number}] score={score:.2f}% elapsed={elapsed:.1f}s")
            return score
        except Exception as e:
            elapsed = time.time() - t0
            record = {
                "trial": trial.number,
                "score": -1.0,
                "elapsed_s": elapsed,
                "params": params,
                "error": str(e),
            }
            with results_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(f"[trial {trial.number}] failed after {elapsed:.1f}s: {e}")
            return -1.0
        finally:
            MAIN_FILE.write_text(original_text)

    try:
        study.optimize(objective, n_trials=args.trials)
    finally:
        MAIN_FILE.write_text(original_text)

    print("\n=== Best Trial ===")
    print(f"score={study.best_value:.2f}%")
    for k, v in study.best_params.items():
        print(f"{k} = {v:.4f}")
    print(f"\nFull trial logs: {results_path}")


if __name__ == "__main__":
    main()
