• Implemented: I changed only the hybrid routing logic in main.py to a validation-first strategy instead of raw confidence gating.

  What changed in generate_hybrid:

  - main.py:97 now does:

  1. Run on-device first.
  2. Infer expected intents from user text (weather, alarm, timer, message, reminder, contacts, play music).
  3. Validate local output against tool schema (required args + type checks).
  4. Validate semantic consistency against extracted slots (time, minutes, location, recipient/message, reminder title/time, contact query, song).
  5. Apply dynamic confidence thresholds by complexity:
      - single-intent: 0.80
      - two intents: 0.92
      - three+ intents: 0.96
  6. Fallback to cloud only if local output fails semantic/schema checks or confidence threshold.

  Why this should improve your benchmark:

  - It explicitly catches the failure mode you showed: high-confidence but wrong local calls (especially timer/reminder/multi-call hard tasks).
  - It preserves local speed/ratio when local calls are actually coherent.

  Validation status:

  - Syntax check passed: python -m py_compile main.py.
  - Full benchmark could not be completed in this environment due blocked network access for Gemini fallback (httpx.ConnectError), so I couldn’t produce a
    new score locally.

  Next step to verify on your side:

  1. Run python benchmark.py with working Gemini connectivity and compare score/F1 breakdown vs your current 50.0%.