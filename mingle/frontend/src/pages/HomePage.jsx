import { useNavigate } from "react-router-dom";

const features = [
  {
    icon: "+",
    title: "Create Profile",
    desc: "Build your digital business card",
    color: "#6366f1",
    gradient: "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
    path: "/create"
  },
  {
    icon: "@",
    title: "My Network",
    desc: "View your saved contacts",
    color: "#10b981",
    gradient: "linear-gradient(135deg, #10b981 0%, #059669 100%)",
    path: "/network"
  },
  {
    icon: "~",
    title: "Find my people",
    desc: "AI-ranked connection discovery",
    color: "#f59e0b",
    gradient: "linear-gradient(135deg, #f59e0b 0%, #d97706 100%)",
    path: "/smart-query"
  },
  {
    icon: "🎤",
    title: "Voice Note",
    desc: "Quick voice-to-email follow-ups",
    color: "#ef4444",
    gradient: "linear-gradient(135deg, #ef4444 0%, #dc2626 100%)",
    path: "/voice-note",
    isNew: true
  }
];

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
      background: "linear-gradient(180deg, #f8fafc 0%, #e0e7ff 100%)",
    }}>
      {/* Logo & Header */}
      <div style={{ textAlign: "center", marginBottom: "48px" }}>
        <div style={{
          width: "80px",
          height: "80px",
          margin: "0 auto 20px",
          borderRadius: "20px",
          background: "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: "36px",
          boxShadow: "0 10px 40px rgba(99, 102, 241, 0.3)",
        }}>
          🤝
        </div>
        <h1 style={{ 
          fontSize: "3.5rem", 
          fontWeight: 800, 
          background: "linear-gradient(135deg, #1e293b 0%, #334155 100%)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          lineHeight: 1.1 
        }}>
          Mingle
        </h1>
        <p style={{ 
          fontSize: "1.1rem", 
          color: "#64748b", 
          marginTop: "16px", 
          maxWidth: "480px",
          lineHeight: 1.6
        }}>
          Smart professional networking powered by on-device AI.
          Exchange cards, discover connections, draft personalised outreach.
        </p>
      </div>

      {/* Feature Cards */}
      <div style={{ 
        display: "grid", 
        gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
        gap: "20px", 
        maxWidth: "960px",
        width: "100%",
        padding: "0 20px"
      }}>
        {features.map((f, i) => (
          <button 
            key={i}
            onClick={() => navigate(f.path)}
            className="card-hover"
            style={{
              position: "relative",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              padding: "32px 24px",
              background: "#fff",
              borderRadius: "20px",
              boxShadow: "0 4px 20px rgba(0,0,0,0.06)",
              cursor: "pointer",
              border: "none",
              textAlign: "center",
              overflow: "hidden",
            }}
          >
            {/* Top accent bar */}
            <div style={{
              position: "absolute",
              top: 0,
              left: 0,
              right: 0,
              height: "4px",
              background: f.gradient,
            }} />

            {/* New badge */}
            {f.isNew && (
              <div style={{
                position: "absolute",
                top: "12px",
                right: "12px",
                padding: "4px 10px",
                background: f.gradient,
                borderRadius: "12px",
                fontSize: "11px",
                fontWeight: 700,
                color: "white",
                textTransform: "uppercase",
                letterSpacing: "0.05em"
              }}>
                New
              </div>
            )}

            {/* Icon */}
            <div style={{
              width: "64px",
              height: "64px",
              borderRadius: "16px",
              background: `${f.color}15`,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: f.icon.length > 1 ? "28px" : "32px",
              color: f.color,
              fontWeight: 700,
              marginBottom: "16px",
            }}>
              {f.icon}
            </div>

            <strong style={{ 
              fontSize: "1.1rem", 
              color: "#1e293b",
              marginBottom: "8px"
            }}>
              {f.title}
            </strong>
            <span style={{ 
              fontSize: "0.9rem", 
              color: "#94a3b8",
              lineHeight: 1.5
            }}>
              {f.desc}
            </span>
          </button>
        ))}
      </div>

      {/* Footer */}
      <div style={{ 
        marginTop: "56px", 
        textAlign: "center",
        color: "#94a3b8",
        fontSize: "13px"
      }}>
        <div style={{
          display: "inline-flex",
          alignItems: "center",
          gap: "8px",
          padding: "8px 16px",
          background: "rgba(255,255,255,0.6)",
          borderRadius: "20px",
          backdropFilter: "blur(8px)"
        }}>
          <span style={{ fontSize: "16px" }}>⚡</span>
          Powered by FunctionGemma on-device inference + Gemini Cloud fallback
        </div>
      </div>
    </div>
  );
}
