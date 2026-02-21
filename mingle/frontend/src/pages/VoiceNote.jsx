import { useState } from "react";
import { useNavigate } from "react-router-dom";

export default function VoiceNote() {
  const navigate = useNavigate();
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [mediaRecorder, setMediaRecorder] = useState(null);

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const chunks = [];

      recorder.ondataavailable = (e) => chunks.push(e.data);
      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: "audio/webm" });
        setAudioBlob(blob);
      };

      recorder.start();
      setMediaRecorder(recorder);
      setIsRecording(true);
      setError(null);
    } catch (err) {
      setError("Failed to access microphone: " + err.message);
    }
  }

  function stopRecording() {
    if (mediaRecorder) {
      mediaRecorder.stop();
      mediaRecorder.stream.getTracks().forEach(track => track.stop());
      setIsRecording(false);
    }
  }

  async function processVoiceNote() {
    if (!audioBlob) return;
    
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("audio", audioBlob, "voice-note.webm");

      const response = await fetch("http://34.67.48.62:3001/api/voicenotes/process", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to process voice note");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setAudioBlob(null);
    setResult(null);
    setError(null);
  }

  return (
    <div style={{ maxWidth: "760px", margin: "0 auto", padding: "40px 20px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "16px", marginBottom: "28px" }}>
        <button onClick={() => navigate("/")} style={backBtn}>← Home</button>
        <h1 style={{ fontSize: "1.8rem", fontWeight: 800 }}>🎤 Voice Note</h1>
      </div>

      <div style={{
        background: "#fff",
        borderRadius: "14px",
        padding: "32px",
        boxShadow: "0 2px 10px rgba(0,0,0,0.07)",
        marginBottom: "24px",
      }}>
        <p style={{ color: "#666", marginBottom: "24px", lineHeight: 1.6 }}>
          Record a voice note like: <em>"Really enjoyed meeting Joe and talking about Cactus. Send an email to schedule a follow-up meeting."</em>
        </p>

        <div style={{ display: "flex", gap: "12px", alignItems: "center", marginBottom: "20px" }}>
          {!isRecording && !audioBlob && (
            <button onClick={startRecording} style={recordBtn}>
              🎙️ Start Recording
            </button>
          )}

          {isRecording && (
            <>
              <button onClick={stopRecording} style={stopBtn}>
                ⏹️ Stop Recording
              </button>
              <span style={{ color: "#e74c3c", fontWeight: 600, animation: "pulse 1.5s infinite" }}>
                ● Recording...
              </span>
            </>
          )}

          {audioBlob && !result && (
            <>
              <button onClick={processVoiceNote} disabled={loading} style={processBtn}>
                {loading ? "Processing..." : "✨ Process Voice Note"}
              </button>
              <button onClick={reset} style={resetBtn}>Reset</button>
            </>
          )}
        </div>

        {error && (
          <div style={{ padding: "12px 16px", background: "#fde8e8", borderRadius: "8px", color: "#c0392b", marginTop: "16px" }}>
            {error}
          </div>
        )}

        {result && (
          <div style={{ marginTop: "24px" }}>
            <div style={{
              padding: "16px",
              background: "#e8f5e9",
              borderRadius: "8px",
              marginBottom: "16px",
            }}>
              <div style={{ fontWeight: 700, color: "#2e7d32", marginBottom: "8px" }}>✅ Processed Successfully!</div>
              <p style={{ margin: "6px 0", fontSize: "0.9rem" }}><strong>Transcript:</strong> {result.transcript}</p>
              <p style={{ margin: "6px 0", fontSize: "0.9rem" }}><strong>Contact:</strong> {result.contact_name}</p>
              <p style={{ margin: "6px 0", fontSize: "0.9rem" }}><strong>Action:</strong> {result.action}</p>
            </div>

            <div style={{
              padding: "20px",
              background: "#f5f5f5",
              borderRadius: "8px",
              border: "1px solid #ddd",
            }}>
              <div style={{ fontWeight: 700, marginBottom: "12px", color: "#444" }}>📧 Email Draft</div>
              <div style={{ fontSize: "0.85rem", color: "#666", marginBottom: "4px" }}>
                To: {result.contact_email}
              </div>
              <pre style={{
                whiteSpace: "pre-wrap",
                fontFamily: "'Georgia', serif",
                fontSize: "0.95rem",
                lineHeight: 1.6,
                color: "#333",
                margin: 0,
              }}>
                {result.email_draft}
              </pre>
            </div>

            <button onClick={reset} style={{ ...resetBtn, marginTop: "16px" }}>
              🎙️ Record Another
            </button>
          </div>
        )}
      </div>

      <div style={{
        padding: "16px",
        background: "#fffbea",
        borderRadius: "8px",
        border: "1px solid #ffd700",
        fontSize: "0.85rem",
        color: "#856404",
      }}>
        <strong>💡 How it works:</strong> Your voice note is transcribed using AI, then Mingle analyzes it to identify the contact and action. An email draft is generated automatically using FunctionGemma (on-device) with Gemini fallback.
      </div>
    </div>
  );
}

const backBtn = {
  padding: "6px 14px",
  background: "none",
  border: "1px solid #ddd",
  borderRadius: "8px",
  cursor: "pointer",
  color: "#555",
  fontSize: "0.85rem",
};

const recordBtn = {
  padding: "12px 24px",
  background: "#e74c3c",
  color: "#fff",
  border: "none",
  borderRadius: "8px",
  fontWeight: 700,
  cursor: "pointer",
  fontSize: "1rem",
};

const stopBtn = {
  padding: "12px 24px",
  background: "#34495e",
  color: "#fff",
  border: "none",
  borderRadius: "8px",
  fontWeight: 700,
  cursor: "pointer",
  fontSize: "1rem",
};

const processBtn = {
  padding: "12px 24px",
  background: "#6c63ff",
  color: "#fff",
  border: "none",
  borderRadius: "8px",
  fontWeight: 700,
  cursor: "pointer",
  fontSize: "1rem",
};

const resetBtn = {
  padding: "8px 16px",
  background: "none",
  border: "1px solid #ddd",
  borderRadius: "8px",
  cursor: "pointer",
  color: "#555",
  fontSize: "0.9rem",
};
