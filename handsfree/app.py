"""
HandsFree — Voice-First Personal Agent
Streamlit app: voice → transcribe (on-device) → location inject → hybrid inference → execute
"""

import sys
import os
import time
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cactus", "python", "src"))

import streamlit as st
from audio_recorder_streamlit import audio_recorder

# ── Local modules ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from handsfree.tools import ALL_TOOLS, TOOL_MAP
from handsfree.location import detect_location_intent
from handsfree.executor import execute
from main import generate_hybrid

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HandsFree",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background: #0f0f0f; }
  .stApp { background: #0f0f0f; color: #f0f0f0; }
  h1 { font-size: 2.4rem !important; font-weight: 800; }
  .pipeline-step {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 14px 18px;
    margin: 6px 0;
    font-size: 0.92rem;
  }
  .pipeline-step.active { border-color: #4ade80; background: #0d2218; }
  .pipeline-step.error  { border-color: #f87171; background: #1f0d0d; }
  .badge-local  { background:#166534; color:#86efac; padding:2px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
  .badge-cloud  { background:#1e3a8a; color:#93c5fd; padding:2px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
  .badge-retry  { background:#713f12; color:#fcd34d; padding:2px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
  .result-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 14px;
    padding: 18px 22px;
    margin: 10px 0;
  }
  .timing-bar {
    background: #111;
    border-radius: 8px;
    height: 8px;
    margin: 4px 0;
    overflow: hidden;
  }
  .timing-fill {
    height: 8px;
    background: linear-gradient(90deg, #4ade80, #22d3ee);
    border-radius: 8px;
  }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def transcribe_audio(wav_bytes: bytes) -> tuple[str, float]:
    """Transcribe audio bytes on-device via cactus_transcribe. Returns (text, ms)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        tmp_path = f.name

    t0 = time.time()
    text = ""
    try:
        # Try Python cactus API first
        from cactus import cactus_transcribe
        text = cactus_transcribe(tmp_path)
    except (ImportError, Exception):
        try:
            # Fallback: run cactus CLI
            import subprocess
            result = subprocess.run(
                ["cactus", "transcribe", "--file", tmp_path],
                capture_output=True, text=True, timeout=30,
            )
            text = result.stdout.strip()
        except Exception:
            text = ""
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    elapsed = (time.time() - t0) * 1000
    return text, elapsed


def source_badge(source: str) -> str:
    if "on-device" in source:
        return '<span class="badge-local">⚡ On-Device</span>'
    elif "retry" in source:
        return '<span class="badge-retry">🔄 On-Device (retry)</span>'
    else:
        return '<span class="badge-cloud">☁️ Cloud</span>'


def render_pipeline(steps: list[dict]):
    """Render a vertical pipeline of steps."""
    st.markdown("#### 🔄 Pipeline")
    for step in steps:
        cls = "active" if step.get("ok") else ("error" if step.get("error") else "")
        icon = "✅" if step.get("ok") else ("❌" if step.get("error") else "⬜")
        detail = f" — {step.get('detail', '')}" if step.get("detail") else ""
        timing = f" <span style='color:#6b7280'>({step['ms']:.0f}ms)</span>" if step.get("ms") else ""
        st.markdown(
            f'<div class="pipeline-step {cls}">{icon} <b>{step["label"]}</b>{detail}{timing}</div>',
            unsafe_allow_html=True,
        )


def render_result(call_result: dict):
    """Render one executed function call result."""
    fn     = call_result["function"]
    args   = call_result["arguments"]
    result = call_result["result"]
    icon   = result.get("icon", "📦")

    st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
    st.markdown(f"**{icon} `{fn}`**")

    arg_str = ", ".join(f"`{k}`: {json.dumps(v)}" for k, v in args.items())
    st.caption(f"Called with: {arg_str}")

    status = result.get("status", "unknown")
    if status == "error":
        st.error(result.get("error", "Unknown error"))

    elif fn == "get_current_location":
        st.markdown(f"**📍 {result.get('address', '')}**")
        c1, c2 = st.columns(2)
        c1.metric("Latitude",  result.get("latitude", ""))
        c2.metric("Longitude", result.get("longitude", ""))
        c1.metric("Source",    result.get("source", ""))
        link = result.get("maps_link", "")
        if link:
            st.markdown(f"[🗺️ Open in Google Maps]({link})")
        if result.get("full_address") and result.get("full_address") != result.get("address"):
            st.caption(f"Full address: {result['full_address']}")

    elif fn == "get_weather":
        c1, c2, c3 = st.columns(3)
        c1.metric("📍 Location",  result.get("location", ""))
        c2.metric("🌡️ Temp",      f"{result.get('temp_f')}°F / {result.get('temp_c')}°C")
        c3.metric("🌤️ Condition", result.get("condition", ""))
        c1.metric("💧 Humidity",  result.get("humidity", ""))
        c2.metric("💨 Wind",      result.get("wind", ""))

    elif fn == "get_directions":
        st.markdown(f"**From:** {result.get('from', '')}")
        st.markdown(f"**To:** {result.get('to', '')}")
        c1, c2 = st.columns(2)
        c1.metric("⏱️ Duration", result.get("duration", ""))
        c2.metric("📏 Distance", result.get("distance", ""))
        steps = result.get("steps", [])
        if steps:
            st.markdown("**Turn-by-turn:**")
            for i, s in enumerate(steps, 1):
                st.markdown(f"{i}. {s}")
        url = result.get("maps_url", "")
        if url:
            st.markdown(f"[🗺️ Open in Google Maps]({url})")

    elif fn in ("find_nearby", "search_along_route"):
        if fn == "search_along_route":
            st.markdown(f"**Route:** {result.get('route', '')}  ({result.get('route_duration','')} · {result.get('route_distance','')})")
        places = result.get("results", [])
        for p in places:
            stars = f"⭐ {p.get('rating', 'N/A')}" if p.get("rating") != "N/A" else ""
            status_badge = p.get("status", "")
            st.markdown(f"- **{p.get('name','')}** {stars}  \n  {p.get('address', '')}  {f'· {status_badge}' if status_badge else ''}")

    else:
        # Generic: show scalar fields, skip internals
        skip = {"status", "icon"}
        for k, v in result.items():
            if k in skip:
                continue
            if k == "maps_url":
                st.markdown(f"[🗺️ Open in Google Maps]({v})")
            elif k == "maps_link":
                st.markdown(f"[📍 View Location]({v})")
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        st.markdown(f"- {' · '.join(str(x) for x in item.values())}")
                    else:
                        st.markdown(f"- {item}")
            else:
                st.markdown(f"**{k.replace('_', ' ').title()}**: {v}")

    st.markdown('</div>', unsafe_allow_html=True)


# ── Main UI ────────────────────────────────────────────────────────────────────

col_header, col_logo = st.columns([5, 1])
with col_header:
    st.markdown("# 🎙️ HandsFree")
    st.markdown("*Voice-first personal agent — on-device speed, cloud intelligence*")

st.divider()

col_input, col_pipeline = st.columns([3, 2])

with col_input:
    st.markdown("### 🎤 Speak or Type a Command")

    # ── Audio recorder ─────────────────────────────────────────────────────
    st.markdown("**Record voice command:**")
    audio_bytes = audio_recorder(
        text="",
        recording_color="#4ade80",
        neutral_color="#374151",
        icon_name="microphone",
        icon_size="2x",
        pause_threshold=2.0,
        sample_rate=16000,
    )

    # ── Text fallback ───────────────────────────────────────────────────────
    st.markdown("**…or type it:**")
    text_input = st.text_input(
        label="command",
        label_visibility="collapsed",
        placeholder="e.g. Send my location to Mom and check weather in SF",
    )

    run_btn = st.button("▶ Run", type="primary", use_container_width=True)

    # ── Example commands ────────────────────────────────────────────────────
    with st.expander("💡 Example commands"):
        examples = [
            "Set an alarm for 7:30 AM",
            "Send my location to Mom",
            "Play Bohemian Rhapsody",
            "Remind me to take medicine at 8:00 PM",
            "Find coffee shops near me and text John saying I'll be late",
            "Set a timer for 15 minutes and check the weather in San Francisco",
            "Get directions from here to Golden Gate Bridge",
            "Search for Tom in my contacts and send him a message saying happy birthday",
        ]
        for ex in examples:
            if st.button(ex, key=ex, use_container_width=True):
                st.session_state["injected_command"] = ex

with col_pipeline:
    pipeline_placeholder = st.empty()
    pipeline_placeholder.markdown("*Pipeline will appear here after running a command.*")


# ── Session state ──────────────────────────────────────────────────────────────
if "injected_command" not in st.session_state:
    st.session_state["injected_command"] = ""

# Prefer injected example over text input
command_text = st.session_state.get("injected_command") or text_input

# ── Run pipeline ───────────────────────────────────────────────────────────────
if (run_btn or st.session_state.get("injected_command")) and (audio_bytes or command_text):

    # Clear injected command after consuming it
    st.session_state["injected_command"] = ""

    steps = []
    final_command = command_text
    timings = {}

    st.divider()
    st.markdown("### ⚡ Running Pipeline…")
    progress = st.progress(0)

    # ── Step 1: Transcription ───────────────────────────────────────────────
    transcription_ms = 0
    if audio_bytes and not command_text:
        with st.spinner("🎙️ Transcribing on-device…"):
            final_command, transcription_ms = transcribe_audio(audio_bytes)
        if not final_command:
            st.error("Transcription returned empty. Please try again or type your command.")
            st.stop()
        steps.append({"label": "Voice → Text (Whisper on-device)", "ok": True,
                       "detail": f'"{final_command[:50]}…"' if len(final_command) > 50 else f'"{final_command}"',
                       "ms": transcription_ms})
    else:
        steps.append({"label": "Voice → Text", "ok": True,
                       "detail": "Text input (no transcription needed)", "ms": 0})
    timings["transcription_ms"] = transcription_ms
    progress.progress(15)

    # Display current transcribed/typed command
    st.markdown(f"**📝 Command:** `{final_command}`")

    # ── Step 2: Location intent detection ───────────────────────────────────
    location_info = None
    location_ms = 0

    if detect_location_intent(final_command):
        # User is asking where they are → let get_current_location tool handle it
        # Do NOT inject GPS into prompt (it would give the model the answer,
        # so it wouldn’t bother calling the tool)
        steps.append({"label": "Location Query Detected", "ok": True,
                       "detail": "Routing to get_current_location", "ms": 1})
    else:
        steps.append({"label": "Location Intent Check", "ok": True,
                       "detail": "No location needed", "ms": 1})

    timings["location_ms"] = location_ms
    progress.progress(35)

    # ── Step 3: Smart routing + inference ───────────────────────────────────
    messages = [{"role": "user", "content": final_command}]
    tools = [
        {k: v for k, v in t.items() if k != "on_device"}
        for t in ALL_TOOLS
    ]

    with st.spinner("🤖 Running hybrid inference…"):
        t0 = time.time()
        inference_result = generate_hybrid(messages, tools)
        inference_ms = (time.time() - t0) * 1000

    source = inference_result.get("source", "unknown")
    fn_calls = inference_result.get("function_calls", [])
    confidence = inference_result.get("confidence", None)

    routing_detail = source
    if confidence is not None:
        routing_detail += f" | conf={confidence:.2f}"

    steps.append({
        "label": f"Hybrid Routing → Inference",
        "ok": bool(fn_calls),
        "error": not bool(fn_calls),
        "detail": routing_detail,
        "ms": inference_ms,
    })
    timings["inference_ms"] = inference_ms
    progress.progress(65)

    # ── Step 4: Execute function calls ──────────────────────────────────────
    if fn_calls:
        t0 = time.time()
        exec_results = execute(fn_calls)
        exec_ms = (time.time() - t0) * 1000
        fn_names = ", ".join(c["function"] for c in exec_results)
        steps.append({"label": "Execute Function Calls", "ok": True,
                       "detail": fn_names, "ms": exec_ms})
        timings["exec_ms"] = exec_ms
    else:
        steps.append({"label": "Execute Function Calls", "error": True,
                       "detail": "No function calls returned"})
        exec_results = []

    progress.progress(100)

    # ── Render pipeline ─────────────────────────────────────────────────────
    with col_pipeline:
        pipeline_placeholder.empty()
        with pipeline_placeholder.container():
            render_pipeline(steps)

            # Timing summary
            total_ms = sum(v for v in timings.values())
            st.markdown("---")
            st.markdown("#### ⏱️ Timing Breakdown")
            for label, ms in {
                "🎙️ Transcription": timings.get("transcription_ms", 0),
                "📍 Location":      timings.get("location_ms", 0),
                "🤖 Inference":     timings.get("inference_ms", 0),
                "⚙️ Execution":     timings.get("exec_ms", 0),
            }.items():
                pct = int((ms / total_ms * 100)) if total_ms > 0 else 0
                st.markdown(f"<small>{label}: **{ms:.0f}ms** ({pct}%)</small>", unsafe_allow_html=True)
                st.markdown(
                    f'<div class="timing-bar"><div class="timing-fill" style="width:{pct}%"></div></div>',
                    unsafe_allow_html=True,
                )
            st.markdown(f"**Total: {total_ms:.0f}ms**")

            # Routing badge
            st.markdown(f"**Routing:** {source_badge(source)}", unsafe_allow_html=True)

    # ── Results ─────────────────────────────────────────────────────────────
    if exec_results:
        st.markdown("### ✅ Results")
        for r in exec_results:
            render_result(r)
    else:
        st.warning("No function calls were generated. Try rephrasing your command.")

    # ── Location info card ───────────────────────────────────────────────────
    if location_info:
        st.markdown("### 📍 Location Used")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Address", location_info["address"])
            st.metric("Source", location_info["source"])
        with c2:
            st.metric("Coordinates", f"{location_info['lat']:.5f}, {location_info['lon']:.5f}")
            st.markdown(f"[View on Maps]({location_info['maps_link']})")

    # ── Raw debug ────────────────────────────────────────────────────────────
    with st.expander("🔍 Raw inference output"):
        st.json(inference_result)


# ── Sidebar: About ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎙️ HandsFree")
    st.markdown("""
**Pipeline stages:**

1. 🎤 Voice capture (browser mic)
2. 🧠 On-device transcription (Whisper via Cactus)
3. 📍 Location intent detection (keyword scan)
4. 🛰️ GPS injection (CoreLocation, no API)
5. ⚡ Hybrid routing (FunctionGemma ↔ Gemini)
6. ✅ Function execution

---

**Available tools:**
""")
    for t in ALL_TOOLS:
        badge = "⚡" if t.get("on_device") else "☁️"
        st.markdown(f"{badge} `{t['name']}`")

    st.markdown("""
---
⚡ = On-device (FunctionGemma)
☁️ = Cloud (Gemini)
""")
