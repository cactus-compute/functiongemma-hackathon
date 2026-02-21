# Install & Setup Notes

## Prerequisites
- Mac (M-series recommended)
- Python 3.12
- Gemini API key with billing enabled

## Steps

### 1. Clone repos
```bash
git clone https://github.com/cactus-compute/functiongemma-hackathon
cd functiongemma-hackathon
git clone https://github.com/cactus-compute/cactus
```

> **Our setup**: We had cactus cloned separately at `~/dev/cactus`, so we symlinked it:
> ```bash
> ln -s /Users/maxime/dev/cactus cactus
> ```
> This works because `main.py` does `sys.path.insert(0, "cactus/python/src")` — it expects `cactus/` as a subdirectory.

### 2. Build cactus
```bash
cd cactus && source ./setup && cd ..
cactus build --python
cactus download google/functiongemma-270m-it --reconvert
```

### 3. Auth
```bash
cactus auth  # enter token from https://cactuscompute.com/dashboard/api-keys
```

### 4. Python deps
```bash
source cactus/venv/bin/activate  # MUST use cactus venv
pip install google-genai
```

### 5. Gemini API key

```bash
export GEMINI_API_KEY='...'
```

**Gotchas we hit:**
- Free tier quota is `limit: 0` — completely blocked. You MUST have billing enabled.
- Claim GCP credits via hackathon links (SF, Online, etc.) — but coupons can be exhausted.
- Credits go to a **billing account**, not a project. You must manually link the billing account to your GCP project:
  - Go to GCP Console → Billing → Your projects → find your project → Change billing → select the trial account with credits.
- The "Default Gemini Project" auto-created by AI Studio has broken permissions. Use a project you created yourself.
- Our working key is in the **"hackathon"** project (`gen-lang-client-0211798491`) linked to trial billing account `01C497-598B90-B47870`.

### 6. Model fix

`gemini-2.0-flash` is deprecated for new users. Changed to `gemini-2.5-flash` in `main.py`:
```python
# main.py line 75
model="gemini-2.5-flash",  # was gemini-2.0-flash
```

### 7. Run benchmark
```bash
source cactus/venv/bin/activate
export GEMINI_API_KEY='...'
python benchmark.py
```

## Baseline Results

```
  TOTAL SCORE: 56.1%

  easy     avg F1=0.70  avg time=392ms   on-device=7/10  cloud=3/10
  medium   avg F1=0.70  avg time=742ms   on-device=4/10  cloud=6/10
  hard     avg F1=0.77  avg time=1033ms  on-device=4/10  cloud=6/10
  overall  avg F1=0.72  avg time=722ms   total time=21.7s
           on-device=15/30 (50%)  cloud=15/30 (50%)
```

Key observations:
- On-device is ~6-8x faster (~170ms vs ~1200ms)
- Some easy cases fail on-device (timer, reminder, search)
- Cloud fallback isn't always correct either (reminder_meeting F1=0, message_among_four F1=0)
- Hard multi-tool cases do better on cloud
