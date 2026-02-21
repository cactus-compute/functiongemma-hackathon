"""
Test harness for iterating on generate_hybrid strategy.
Runs benchmark cases and shows detailed diagnostics without cloud calls.

Usage:
    python test_harness.py                  # Run all, show summary
    python test_harness.py --verbose        # Show per-case details
    python test_harness.py --local-only     # Skip cloud, show what local produces
    python test_harness.py --case timer_5min  # Run single case
"""

import sys, os, json, time, argparse
sys.path.insert(0, "cactus/python/src")
os.environ["CACTUS_NO_CLOUD_TELE"] = "1"

from benchmark import BENCHMARKS, compute_f1, compute_total_score
from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset

MODEL_PATH = "cactus/weights/functiongemma-270m-it"


def run_local_diagnostic(case, model):
    """Run a case through FunctionGemma and return detailed info."""
    cactus_reset(model)

    cactus_tools = [{"type": "function", "function": t} for t in case["tools"]]
    messages = [
        {"role": "system", "content": "You are a helpful assistant that can use tools."}
    ] + case["messages"]

    raw_str = cactus_complete(
        model, messages, tools=cactus_tools,
        force_tools=True, max_tokens=256, temperature=0.0,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {"function_calls": [], "confidence": 0, "total_time_ms": 0, "raw": raw_str, "parse_error": True}

    return {
        "function_calls": raw.get("function_calls", []),
        "confidence": raw.get("confidence", 0),
        "total_time_ms": raw.get("total_time_ms", 0),
        "cloud_handoff": raw.get("cloud_handoff", False),
        "raw_response": raw.get("response", ""),
        "parse_error": False,
    }


def validate_call(call, tools):
    """Check if a function call is valid against tool definitions."""
    tool_names = {t["name"] for t in tools}
    if call["name"] not in tool_names:
        return False, f"unknown function: {call['name']}"

    tool = next(t for t in tools if t["name"] == call["name"])
    required = tool["parameters"].get("required", [])
    args = call.get("arguments", {})
    missing = [r for r in required if r not in args]
    if missing:
        return False, f"missing required args: {missing}"

    return True, "ok"


def analyze_routing_decision(case, local_result):
    """Suggest whether this case should go local or cloud."""
    num_tools = len(case["tools"])
    num_expected = len(case["expected_calls"])
    local_f1 = compute_f1(local_result["function_calls"], case["expected_calls"])

    reasons = []

    # FunctionGemma can't multi-call
    if num_expected > 1:
        reasons.append(f"needs {num_expected} calls (FG can't multi-call)")
        return "cloud", reasons, local_f1

    # Validate the local output
    for call in local_result["function_calls"]:
        valid, msg = validate_call(call, case["tools"])
        if not valid:
            reasons.append(f"invalid call: {msg}")
            return "cloud", reasons, local_f1

    if not local_result["function_calls"]:
        reasons.append("no function calls returned")
        return "cloud", reasons, local_f1

    if local_f1 == 1.0:
        reasons.append("local got it right")
        return "local", reasons, local_f1

    reasons.append(f"local F1={local_f1:.2f}")
    return "cloud", reasons, local_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--case", type=str, help="Run single case by name")
    args = parser.parse_args()

    cases = BENCHMARKS
    if args.case:
        cases = [c for c in BENCHMARKS if c["name"] == args.case]
        if not cases:
            print(f"Case '{args.case}' not found. Available: {[c['name'] for c in BENCHMARKS]}")
            return

    model = cactus_init(MODEL_PATH)

    results_summary = []
    for i, case in enumerate(cases, 1):
        local = run_local_diagnostic(case, model)
        local_f1 = compute_f1(local["function_calls"], case["expected_calls"])
        suggestion, reasons, _ = analyze_routing_decision(case, local)

        status = "✅" if local_f1 == 1.0 else "⚠️" if local_f1 > 0 else "❌"

        results_summary.append({
            "name": case["name"],
            "difficulty": case["difficulty"],
            "num_tools": len(case["tools"]),
            "num_expected": len(case["expected_calls"]),
            "local_f1": local_f1,
            "confidence": local["confidence"],
            "suggestion": suggestion,
            "reasons": reasons,
            "local_calls": local["function_calls"],
            "expected_calls": case["expected_calls"],
        })

        if args.verbose or args.case:
            print(f"\n[{i}/{len(cases)}] {status} {case['name']} ({case['difficulty']})")
            print(f"  tools={len(case['tools'])} expected_calls={len(case['expected_calls'])}")
            print(f"  local_f1={local_f1:.2f} conf={local['confidence']:.3f} time={local['total_time_ms']:.0f}ms")
            print(f"  expected: {[(c['name'], c['arguments']) for c in case['expected_calls']]}")
            print(f"  got:      {[(c['name'], c.get('arguments',{})) for c in local['function_calls']]}")
            print(f"  suggest:  {suggestion} ({', '.join(reasons)})")
        else:
            print(f"  {status} {case['name']:30s} {case['difficulty']:6s} | local_f1={local_f1:.2f} conf={local['confidence']:.3f} | → {suggestion:5s} | {', '.join(reasons)}")

    cactus_destroy(model)

    # Summary
    print(f"\n{'='*60}")
    print("ROUTING SUMMARY")
    print(f"{'='*60}")

    for diff in ["easy", "medium", "hard"]:
        group = [r for r in results_summary if r["difficulty"] == diff]
        local_correct = sum(1 for r in group if r["local_f1"] == 1.0)
        suggest_local = sum(1 for r in group if r["suggestion"] == "local")
        print(f"  {diff:8s}: {local_correct}/{len(group)} correct locally, {suggest_local}/{len(group)} suggested local")

    total_local_correct = sum(1 for r in results_summary if r["local_f1"] == 1.0)
    total_suggest_local = sum(1 for r in results_summary if r["suggestion"] == "local")
    print(f"  {'total':8s}: {total_local_correct}/{len(results_summary)} correct locally, {total_suggest_local}/{len(results_summary)} suggested local")


if __name__ == "__main__":
    main()
