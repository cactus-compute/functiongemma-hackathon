import { useState } from "react";
import { draftOutreach } from "../api/client.js";

export default function OutreachModal({ sender, recipient, context, onClose }) {
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [source, setSource] = useState(null);

  async function handleDraft() {
    setLoading(true);
    setError(null);
    try {
      const res = await draftOutreach({ sender, recipient, context: context || "" });
      setMessage(res.message);
      setSource(res.source);
    } catch (err) {
      setError(err.response?.data?.error || err.message);
    } finally {
      setLoading(false);
    }
  }

  if (!sender || !recipient) return null;

  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed", inset: 0,
        background: "rgba(0,0,0,0.5)",
        display: "flex", alignItems: "center", justifyContent: "center",
        zIndex: 1000,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: "#fff",
          borderRadius: "16px",
          padding: "28px",
          maxWidth: "520px",
          width: "92%",
        }}
      >
        <h3 style={{ marginBottom: "8px" }}>Draft outreach message</h3>
        <p style={{ fontSize: "0.85rem", color: "#666", marginBottom: "16px" }}>
          From <strong>{sender.name}</strong> to <strong>{recipient.name}</strong>
        </p>

        {!message && (
          <button
            onClick={handleDraft}
            disabled={loading}
            style={{
              padding: "10px 24px",
              background: "#6c63ff", color: "#fff",
              border: "none", borderRadius: "8px", cursor: "pointer",
              opacity: loading ? 0.6 : 1,
            }}
          >
            {loading ? "Drafting…" : "Generate with AI"}
          </button>
        )}

        {error && <p style={{ color: "red", marginTop: "12px", fontSize: "0.85rem" }}>{error}</p>}

        {message && (
          <div style={{ marginTop: "16px" }}>
            <textarea
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              rows={6}
              style={{
                width: "100%", padding: "12px",
                border: "1px solid #ddd", borderRadius: "8px",
                fontSize: "0.9rem", lineHeight: 1.5, resize: "vertical",
              }}
            />
            {source && (
              <p style={{ fontSize: "0.75rem", color: "#999", marginTop: "4px" }}>
                Generated via: {source}
              </p>
            )}
            <button
              onClick={() => navigator.clipboard.writeText(message)}
              style={{
                marginTop: "10px", marginRight: "8px",
                padding: "8px 18px",
                background: "#e8f4ff", color: "#0066cc",
                border: "none", borderRadius: "8px", cursor: "pointer",
              }}
            >
              Copy
            </button>
            <button
              onClick={handleDraft}
              disabled={loading}
              style={{
                padding: "8px 18px",
                background: "#f5f5f5", color: "#333",
                border: "none", borderRadius: "8px", cursor: "pointer",
              }}
            >
              Regenerate
            </button>
          </div>
        )}

        <button
          onClick={onClose}
          style={{
            display: "block", marginTop: "16px",
            padding: "8px 20px",
            background: "none", color: "#888",
            border: "1px solid #ddd", borderRadius: "8px", cursor: "pointer",
          }}
        >
          Close
        </button>
      </div>
    </div>
  );
}
