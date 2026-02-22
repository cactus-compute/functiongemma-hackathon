# Benchmark Metrics — Explained

This document explains all the metrics used in `benchmark.py` to evaluate hybrid on-device/cloud function calling.

---

## 1. F1 Score (60% of total score)

### What It Measures

F1 is a standard metric that combines **precision** and **recall** into a single number between 0 and 1. In this codebase, it measures: **did the model call the right functions with the right arguments?**

```python
precision = matched / len(predicted_calls)   # of what you predicted, how many were correct?
recall    = matched / len(expected_calls)     # of what was expected, how many did you find?
f1        = 2 * precision * recall / (precision + recall)  # harmonic mean of both
```

### Why Not Just Use Accuracy?

Because for the **hard** benchmark cases, the model needs to make **multiple** function calls. Simple accuracy doesn't capture partial credit. F1 does.

### Concrete Examples

#### Perfect score (F1 = 1.0)

User: *"Text Emma good night and check weather in Chicago"*

| Expected | Predicted |
|---|---|
| `send_message(Emma, good night)` | `send_message(Emma, good night)` ✅ |
| `get_weather(Chicago)` | `get_weather(Chicago)` ✅ |

- Matched: 2, Predicted: 2, Expected: 2
- Precision = 2/2 = 1.0, Recall = 2/2 = 1.0
- **F1 = 1.0**

#### Model misses one (F1 = 0.67)

Same prompt, but model only outputs `send_message`:

| Expected | Predicted |
|---|---|
| `send_message(Emma, good night)` | `send_message(Emma, good night)` ✅ |
| `get_weather(Chicago)` | *(missing)* |

- Matched: 1, Predicted: 1, Expected: 2
- Precision = 1/1 = 1.0 (everything it predicted was correct)
- Recall = 1/2 = 0.5 (it missed half the expected calls)
- **F1 = 2 × 1.0 × 0.5 / (1.0 + 0.5) = 0.67**

#### Model hallucinates an extra call (F1 = 0.80)

Model correctly gets both, but also adds a bogus `set_alarm`:

- Matched: 2, Predicted: 3, Expected: 2
- Precision = 2/3 = 0.67 (one prediction was wrong)
- Recall = 2/2 = 1.0 (found everything expected)
- **F1 = 2 × 0.67 × 1.0 / (0.67 + 1.0) = 0.80**

#### Completely wrong (F1 = 0.0)

Model calls `play_music("jazz")` when it should have called `get_weather("Chicago")`:

- Matched: 0
- **F1 = 0.0**

### How Matching Works

The `_call_matches` function in `benchmark.py` checks two things:

1. **Function name must match exactly** — `get_weather` == `get_weather`
2. **All expected argument values must be present** — compared case-insensitively after stripping whitespace

```python
def _normalize(v):
    if isinstance(v, str):
        return v.strip().lower()
    return v
```

So `"San Francisco"` matches `"san francisco"` and `" San Francisco "`, but `"SF"` would **not** match.

### Summary Table

| Situation | Precision | Recall | F1 |
|---|---|---|---|
| Perfect | 1.0 | 1.0 | **1.0** |
| Missed a call | High | Low | **Medium** |
| Extra bogus call | Low | High | **Medium** |
| All wrong | 0 | 0 | **0.0** |

F1 penalizes the model for **both** missing calls it should have made **and** making calls it shouldn't have. That's why it's the right metric here — especially for hard cases where the model needs to make 2–3 calls from a single user message.

---

## 2. Time Score (15% of total score)

### What It Measures

How fast inference is. Faster = better, with a **500ms baseline** as the ceiling.

```python
time_baseline_ms = 500
time_score = max(0, 1 - avg_time / time_baseline_ms)
```

### How It Scales

It's a linear penalty — the closer to 500ms, the closer to 0:

| Avg Latency | Time Score |
|---|---|
| 0ms | 1.0 |
| 100ms | 0.80 |
| 250ms | 0.50 |
| 400ms | 0.20 |
| 500ms+ | 0.0 |

Anything over 500ms is clamped to 0 by `max(0, ...)`. There's no extra penalty for being slow — you just get zero.

### Why It Matters

On-device inference (Cactus) is typically very fast. Cloud fallback (Gemini API) adds network latency. This metric rewards staying on-device and penalizes cloud round-trips.

---

## 3. On-Device Ratio (25% of total score)

### What It Measures

What fraction of benchmark cases were handled entirely on-device (by Cactus/FunctionGemma) vs falling back to Gemini cloud.

```python
on_device_ratio = sum(1 for r in group if r["source"] == "on-device") / len(group)
```

### How It's Determined

In `generate_hybrid`, if the local model's **confidence** meets the threshold, the result is tagged `"on-device"`. Otherwise it falls back to cloud:

```python
def generate_hybrid(messages, tools, confidence_threshold=0.99):
    local = generate_cactus(messages, tools)
    if local["confidence"] >= confidence_threshold:
        local["source"] = "on-device"
        return local
    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    return cloud
```

### Examples

| Cases on-device | Total cases | Ratio |
|---|---|---|
| 10 | 10 | 1.0 (perfect — never hit the cloud) |
| 7 | 10 | 0.7 |
| 0 | 10 | 0.0 (always fell back to cloud) |

### Why It Matters

The whole point of the hybrid architecture is to avoid cloud calls when possible. Cloud calls cost money, add latency, and require connectivity. This metric directly rewards keeping inference local.

---

## 4. Total Score (the composite)

### What It Measures

A single 0–100% number that combines all three metrics above, weighted by difficulty level.

### Formula

For each difficulty level:

```python
level_score = (0.60 * avg_f1) + (0.15 * time_score) + (0.25 * on_device_ratio)
```

Then weighted across difficulty:

```python
difficulty_weights = {"easy": 0.20, "medium": 0.30, "hard": 0.50}
total_score = sum(weight * level_score for each difficulty)
```

### Difficulty Weights

| Difficulty | Weight | Why |
|---|---|---|
| Easy | 20% | 1 tool, straightforward — should be a gimme |
| Medium | 30% | 2–5 tools, must pick the right one |
| Hard | 50% | 2–5 tools, must make **multiple** correct calls |

Hard cases dominate the score. Getting multi-tool calls right is what matters most.

### What a Perfect Score Looks Like

- F1 = 1.0 across all difficulties
- All inferences under 500ms
- All inferences handled on-device
- **Total Score = 100%**

### What Drags the Score Down

| Problem | Impact |
|---|---|
| Wrong/missing function calls | Biggest hit (F1 is 60% of level score) |
| Always falling back to cloud | 25% penalty per level |
| Slow inference (>500ms) | 15% penalty per level |
| Failing on hard cases specifically | Amplified by 50% difficulty weight |
