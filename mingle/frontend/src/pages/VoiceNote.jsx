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
  const [sending, setSending] = useState(false);
  const [sent, setSent] = useState(false);

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
      setError("Microphone access denied. Try the demo instead!");
    }
  }

  function stopRecording() {
    if (mediaRecorder) {
      mediaRecorder.stop();
      mediaRecorder.stream.getTracks().forEach(track => track.stop());
      setIsRecording(false);
    }
  }

  async function loadDemo() {
    setLoading(true);
    setError(null);
    setResult(null);
    setSent(false);
    
    // Simulate processing delay
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    // Demo result for Siddhi
    setResult({
      transcript: "Really enjoyed meeting Siddhi. Send a follow-up email to arrange for a meeting to discuss building networking agents.",
      contact_name: "Siddhi Sharma",
      contact_role: "AI Researcher",
      contact_company: "Google DeepMind",
      contact_email: "siddhi@deepmind.google.com",
      email_subject: "Great meeting you - Let's discuss networking agents",
      email_body: `Hi Siddhi,

It was wonderful meeting you! I really enjoyed our conversation and I'm excited about the potential to collaborate on building networking agents.

I'd love to schedule a follow-up meeting to dive deeper into this topic. Would you have any availability next week for a 30-minute call?

Looking forward to connecting again!

Best regards`,
      tool_calls: [
        { tool: "extract_intent", source: "FunctionGemma (on-device)" },
        { tool: "lookup_contact", source: "Mingle DB" },
        { tool: "draft_email", source: "Gemini Cloud" }
      ],
      source: "demo mode — FunctionGemma + Gemini hybrid"
    });
    
    setLoading(false);
  }

  async function processVoiceNote() {
    if (!audioBlob) return;
    
    setLoading(true);
    setError(null);
    setResult(null);
    setSent(false);

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

  async function handleSendEmail() {
    setSending(true);
    // Simulate sending
    await new Promise(resolve => setTimeout(resolve, 1800));
    setSending(false);
    setSent(true);
  }

  function reset() {
    setAudioBlob(null);
    setResult(null);
    setError(null);
    setSent(false);
    setSending(false);
  }

  return (
    <div style={{ maxWidth: "800px", margin: "0 auto", padding: "40px 20px" }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", gap: "16px", marginBottom: "32px" }}>
        <button onClick={() => navigate("/")} style={backBtn}>← Home</button>
        <h1 style={{ fontSize: "2rem", fontWeight: 800 }}>
          <span style={{ marginRight: "12px" }}>🎤</span>
          <span className="gradient-text">Voice Note</span>
        </h1>
      </div>

      {/* Recording Card */}
      <div style={cardStyle}>
        <p style={{ color: "#64748b", marginBottom: "24px", lineHeight: 1.7, fontSize: "15px" }}>
          Record a quick voice memo after meeting someone. Mingle will identify the contact, 
          extract key details, and draft a follow-up email automatically.
        </p>

        <div style={{ 
          background: "#f8fafc", 
          borderRadius: "12px", 
          padding: "20px", 
          marginBottom: "24px",
          border: "1px dashed #e2e8f0"
        }}>
          <p style={{ fontSize: "14px", color: "#94a3b8", marginBottom: "8px" }}>Example:</p>
          <p style={{ fontStyle: "italic", color: "#475569", lineHeight: 1.6 }}>
            "Really enjoyed meeting Siddhi. Send a follow-up email to arrange for a meeting to discuss building networking agents."
          </p>
        </div>

        <div style={{ display: "flex", gap: "12px", alignItems: "center", flexWrap: "wrap" }}>
          {!isRecording && !audioBlob && !loading && !result && (
            <>
              <button onClick={startRecording} style={recordBtn}>
                <span style={{ fontSize: "20px", marginRight: "8px" }}>🎙️</span>
                Start Recording
              </button>
              <button onClick={loadDemo} style={demoBtn}>
                <span style={{ fontSize: "18px", marginRight: "8px" }}>✨</span>
                Try Demo
              </button>
            </>
          )}

          {loading && (
            <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
              <div className="spinner" style={{
                width: "24px",
                height: "24px",
                border: "3px solid #e2e8f0",
                borderTopColor: "#6366f1",
                borderRadius: "50%",
                animation: "spin 0.8s linear infinite"
              }}></div>
              <span style={{ color: "#6366f1", fontWeight: 600 }}>Processing with AI...</span>
            </div>
          )}

          {isRecording && (
            <>
              <button onClick={stopRecording} style={stopBtn} className="recording-btn">
                <span style={{ fontSize: "20px", marginRight: "8px" }}>⏹️</span>
                Stop Recording
              </button>
              <div className="recording-indicator" style={{ 
                display: "flex", 
                alignItems: "center", 
                gap: "8px",
                color: "#dc2626",
                fontWeight: 600
              }}>
                <span className="pulse-dot" style={{ 
                  width: "12px", 
                  height: "12px", 
                  background: "#dc2626", 
                  borderRadius: "50%",
                  display: "inline-block",
                  animation: "pulse 1s infinite"
                }}></span>
                Recording...
              </div>
            </>
          )}

          {audioBlob && !result && !loading && (
            <>
              <button onClick={processVoiceNote} disabled={loading} style={processBtn}>
                <span style={{ marginRight: "8px" }}>✨</span>
                Process with AI
              </button>
              <button onClick={reset} style={resetBtn}>Cancel</button>
            </>
          )}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="fade-in" style={errorStyle}>
          <span style={{ marginRight: "8px" }}>⚠️</span>
          {error}
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="fade-in">
          {/* Transcript */}
          {result.transcript && (
            <div style={{ 
              background: "#f1f5f9", 
              borderRadius: "12px", 
              padding: "16px 20px", 
              marginBottom: "20px",
              fontStyle: "italic",
              color: "#475569"
            }}>
              "{result.transcript}"
            </div>
          )}

          {/* AI Pipeline Steps */}
          <div style={{ marginBottom: "24px" }}>
            <h3 style={{ fontSize: "14px", fontWeight: 600, color: "#64748b", marginBottom: "12px", textTransform: "uppercase", letterSpacing: "0.05em" }}>
              AI Pipeline
            </h3>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
              {result.tool_calls?.map((tc, i) => (
                <div key={i} className="tool-badge">
                  <span>{tc.tool === "extract_intent" ? "🧠" : tc.tool === "lookup_contact" ? "🔍" : "✉️"}</span>
                  {tc.tool.replace(/_/g, " ")}
                  <span style={{ color: "#6366f1", fontSize: "10px" }}>({tc.source?.split(" ")[0]})</span>
                </div>
              ))}
            </div>
          </div>

          {/* Contact Found Card */}
          {result.contact_name && (
            <div style={contactCardStyle}>
              <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
                <div style={avatarStyle}>
                  {result.contact_name?.charAt(0)}
                </div>
                <div>
                  <h3 style={{ fontSize: "18px", fontWeight: 700, color: "#1e293b", marginBottom: "4px" }}>
                    {result.contact_name}
                  </h3>
                  <p style={{ fontSize: "14px", color: "#64748b" }}>
                    {result.contact_role} @ {result.contact_company}
                  </p>
                  <p style={{ fontSize: "13px", color: "#6366f1", marginTop: "4px" }}>
                    📧 {result.contact_email}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Email Preview */}
          {result.email_body && (
            <div className="email-preview" style={{ marginTop: "20px" }}>
              <div className="email-header">
                <div style={{ fontSize: "12px", opacity: 0.8, marginBottom: "4px" }}>TO: {result.contact_email}</div>
                <div style={{ fontSize: "16px", fontWeight: 600 }}>📧 {result.email_subject}</div>
              </div>
              <div className="email-body">
                <pre style={{ 
                  whiteSpace: "pre-wrap", 
                  fontFamily: "inherit",
                  margin: 0,
                  fontSize: "15px"
                }}>
                  {result.email_body}
                </pre>
              </div>
              <div style={{ 
                padding: "16px 24px", 
                borderTop: "1px solid #e2e8f0",
                display: "flex",
                gap: "12px",
                position: "relative",
                overflow: "hidden"
              }}>
                {sent ? (
                  <div style={{ 
                    display: "flex", 
                    alignItems: "center", 
                    gap: "8px",
                    color: "#10b981",
                    fontWeight: 600
                  }}>
                    <span style={{ fontSize: "20px" }}>✅</span>
                    Email Sent!
                  </div>
                ) : (
                  <>
                    <button onClick={handleSendEmail} disabled={sending} style={{
                      ...sendBtn,
                      opacity: sending ? 0.7 : 1,
                      position: "relative",
                      overflow: "hidden"
                    }}>
                      {sending ? (
                        <>
                          <span className="flying-email" style={{
                            position: "absolute",
                            animation: "flyAway 1.8s ease-in-out forwards",
                            fontSize: "20px"
                          }}>✉️</span>
                          <span style={{ marginLeft: "24px" }}>Sending...</span>
                        </>
                      ) : (
                        <>
                          <span style={{ marginRight: "6px" }}>📤</span>
                          Send Email
                        </>
                      )}
                    </button>
                    <button style={editBtn} disabled={sending}>
                      <span style={{ marginRight: "6px" }}>✏️</span>
                      Edit Draft
                    </button>
                  </>
                )}
              </div>
            </div>
          )}

          {/* Source Badge */}
          <div style={{ marginTop: "20px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div style={{ 
              fontSize: "12px", 
              color: "#94a3b8",
              display: "flex",
              alignItems: "center",
              gap: "6px"
            }}>
              <span style={{ 
                width: "8px", 
                height: "8px", 
                background: result.source?.includes("demo") ? "#f59e0b" : "#10b981",
                borderRadius: "50%"
              }}></span>
              {result.source}
            </div>
            <button onClick={reset} style={resetBtn}>
              🎙️ Record Another
            </button>
          </div>
        </div>
      )}

      {/* Info Box */}
      {!result && !loading && (
        <div style={infoBoxStyle}>
          <strong>💡 How it works:</strong>
          <ol style={{ marginTop: "12px", paddingLeft: "20px", lineHeight: 1.8 }}>
            <li><strong>Extract Intent</strong> — AI identifies the contact name and action</li>
            <li><strong>Lookup Contact</strong> — Searches your Mingle network database</li>
            <li><strong>Draft Email</strong> — Generates a personalized follow-up message</li>
          </ol>
          <p style={{ marginTop: "12px", fontSize: "13px", color: "#6b7280" }}>
            Powered by FunctionGemma (on-device) with Gemini Cloud fallback
          </p>
        </div>
      )}

      {/* CSS Animations */}
      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.5; transform: scale(1.2); }
        }
        @keyframes flyAway {
          0% { transform: translateX(0) translateY(0) rotate(0deg); opacity: 1; }
          30% { transform: translateX(10px) translateY(-15px) rotate(-10deg); opacity: 1; }
          100% { transform: translateX(200px) translateY(-80px) rotate(-30deg) scale(0.5); opacity: 0; }
        }
      `}</style>
    </div>
  );
}

const cardStyle = {
  background: "#fff",
  borderRadius: "16px",
  padding: "32px",
  boxShadow: "0 4px 20px rgba(0,0,0,0.06)",
  marginBottom: "24px",
};

const backBtn = {
  padding: "8px 16px",
  background: "#f1f5f9",
  border: "none",
  borderRadius: "8px",
  cursor: "pointer",
  color: "#475569",
  fontSize: "14px",
  fontWeight: 500,
  transition: "all 0.2s",
};

const recordBtn = {
  padding: "14px 28px",
  background: "linear-gradient(135deg, #dc2626 0%, #b91c1c 100%)",
  color: "#fff",
  border: "none",
  borderRadius: "12px",
  fontWeight: 600,
  cursor: "pointer",
  fontSize: "15px",
  display: "flex",
  alignItems: "center",
  transition: "all 0.2s",
};

const demoBtn = {
  padding: "14px 28px",
  background: "linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)",
  color: "#fff",
  border: "none",
  borderRadius: "12px",
  fontWeight: 600,
  cursor: "pointer",
  fontSize: "15px",
  display: "flex",
  alignItems: "center",
  transition: "all 0.2s",
};

const stopBtn = {
  padding: "14px 28px",
  background: "linear-gradient(135deg, #374151 0%, #1f2937 100%)",
  color: "#fff",
  border: "none",
  borderRadius: "12px",
  fontWeight: 600,
  cursor: "pointer",
  fontSize: "15px",
  display: "flex",
  alignItems: "center",
};

const processBtn = {
  padding: "14px 28px",
  background: "linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)",
  color: "#fff",
  border: "none",
  borderRadius: "12px",
  fontWeight: 600,
  cursor: "pointer",
  fontSize: "15px",
  display: "flex",
  alignItems: "center",
  transition: "all 0.2s",
};

const resetBtn = {
  padding: "10px 20px",
  background: "#f1f5f9",
  border: "none",
  borderRadius: "8px",
  cursor: "pointer",
  color: "#64748b",
  fontSize: "14px",
  fontWeight: 500,
};

const errorStyle = {
  padding: "16px 20px",
  background: "#fef2f2",
  borderRadius: "12px",
  color: "#dc2626",
  marginBottom: "24px",
  fontSize: "14px",
};

const contactCardStyle = {
  background: "linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)",
  borderRadius: "16px",
  padding: "24px",
  border: "1px solid #bbf7d0",
};

const avatarStyle = {
  width: "56px",
  height: "56px",
  borderRadius: "14px",
  background: "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
  color: "white",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  fontSize: "24px",
  fontWeight: 700,
};

const sendBtn = {
  padding: "10px 20px",
  background: "linear-gradient(135deg, #10b981 0%, #059669 100%)",
  color: "white",
  border: "none",
  borderRadius: "8px",
  fontWeight: 600,
  cursor: "pointer",
  fontSize: "14px",
  display: "flex",
  alignItems: "center",
};

const editBtn = {
  padding: "10px 20px",
  background: "white",
  color: "#475569",
  border: "1px solid #e2e8f0",
  borderRadius: "8px",
  fontWeight: 500,
  cursor: "pointer",
  fontSize: "14px",
  display: "flex",
  alignItems: "center",
};

const infoBoxStyle = {
  padding: "24px",
  background: "linear-gradient(135deg, #fefce8 0%, #fef9c3 100%)",
  borderRadius: "16px",
  border: "1px solid #fde047",
  fontSize: "14px",
  color: "#713f12",
  lineHeight: 1.6,
};
