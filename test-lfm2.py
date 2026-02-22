#!/usr/bin/env python3
"""
Test LFM2-VL-450M as a prompt rewriter.

Can it produce useful paraphrases we could pipe into FunctionGemma?

Usage:
    source cactus/venv/bin/activate
    python test-lfm2.py "Set an alarm for 7am"
"""

import sys
import json

sys.path.insert(0, "cactus/python/src")

from cactus import cactus_init, cactus_complete, cactus_destroy

MODEL_PATH = "cactus/weights/lfm2-vl-450m"

SYSTEM_PROMPTS = [
    "You paraphrase user commands. Output only the reworded command, nothing else.",
    "Rewrite the user's message using different words. Keep the same meaning. Be brief.",
    "You are a paraphraser. Say the same thing differently in one sentence.",
]

USER_TEMPLATES = [
    "{prompt}",
    'Paraphrase: "{prompt}"',
]


def test_rewrite(user_prompt):
    model = cactus_init(MODEL_PATH)

    for sys_prompt in SYSTEM_PROMPTS:
        for user_tmpl in USER_TEMPLATES:
            filled = user_tmpl.format(prompt=user_prompt)
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": filled},
            ]

            raw = cactus_complete(
                model,
                messages,
                max_tokens=40,
                stop_sequences=["<|im_end|>", "<end_of_turn>", "\n"],
            )
            parsed = json.loads(raw)

            resp = parsed.get("response", "")
            ms = parsed.get("total_time_ms", "?")
            print(f"System:   {sys_prompt[:60]}...")
            print(f"User:     {filled!r}")
            print(f"Response: {resp!r}")
            print(f"Time:     {ms}ms")
            print("---")

    cactus_destroy(model)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python test-lfm2.py "your prompt"')
        sys.exit(1)

    prompt = " ".join(sys.argv[1:])
    print(f"\nOriginal: {prompt!r}\n")
    test_rewrite(prompt)
