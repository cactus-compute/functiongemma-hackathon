import { useNavigate } from "react-router-dom";

const ctaStyle = {
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  padding: "32px 24px",
  background: "#fff",
  borderRadius: "16px",
  boxShadow: "0 2px 12px rgba(0,0,0,0.08)",
  cursor: "pointer",
  flex: 1,
  minWidth: "200px",
  maxWidth: "260px",
  transition: "transform 0.15s ease, box-shadow 0.15s ease",
  textAlign: "center",
  border: "none",
  color: "inherit",
  textDecoration: "none",
};

export default function HomePage() {
  const navigate = useNavigate();

  return (
    <div style={{
      minHeight: "100vh",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      padding: "40px 20px",
      background: "linear-gradient(135deg, #f8f9fa 0%, #e8f4ff 100%)",
    }}>
      <div style={{ textAlign: "center", marginBottom: "48px" }}>
        <h1 style={{ fontSize: "3rem", fontWeight: 800, color: "#1a1a2e", lineHeight: 1.1 }}>
          Mingle
        </h1>
        <p style={{ fontSize: "1.15rem", color: "#555", marginTop: "12px", maxWidth: "460px" }}>
          Smart professional networking powered by on-device AI.
          Exchange cards, discover connections, draft personalised outreach.
        </p>
      </div>

      <div style={{ display: "flex", gap: "20px", flexWrap: "wrap", justifyContent: "center" }}>
        <button style={{ ...ctaStyle, borderTop: "4px solid #6c63ff" }} onClick={() => navigate("/create")}>
          <span style={{ fontSize: "2.5rem" }}>+</span>
          <strong style={{ marginTop: "12px", fontSize: "1rem" }}>Create Profile</strong>
          <span style={{ fontSize: "0.85rem", color: "#888", marginTop: "6px" }}>
            Build your digital business card
          </span>
        </button>

        <button style={{ ...ctaStyle, borderTop: "4px solid #2ecc71" }} onClick={() => navigate("/network")}>
          <span style={{ fontSize: "2.5rem" }}>@</span>
          <strong style={{ marginTop: "12px", fontSize: "1rem" }}>My Network</strong>
          <span style={{ fontSize: "0.85rem", color: "#888", marginTop: "6px" }}>
            View your saved contacts
          </span>
        </button>

        <button style={{ ...ctaStyle, borderTop: "4px solid #f39c12" }} onClick={() => navigate("/smart-query")}>
          <span style={{ fontSize: "2.5rem" }}>~</span>
          <strong style={{ marginTop: "12px", fontSize: "1rem" }}>Smart Query</strong>
          <span style={{ fontSize: "0.85rem", color: "#888", marginTop: "6px" }}>
            AI-ranked connection discovery
          </span>
        </button>
      </div>

      <p style={{ marginTop: "48px", fontSize: "0.8rem", color: "#aaa" }}>
        Powered by FunctionGemma on-device inference + Gemini Cloud fallback
      </p>
    </div>
  );
}
