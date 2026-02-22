import { useState } from "react";
import { useNavigate } from "react-router-dom";

const LOOKING_FOR_OPTIONS = ["Co-founder", "Collaborators", "Mentorship", "Investors", "Customers", "Employees", "Advice"];
const DOMAIN_OPTIONS = ["AI/ML", "Hardware", "FinTech", "HealthTech", "EdTech", "Climate", "Web3", "Enterprise SaaS", "Consumer", "Developer Tools"];
const HELP_OPTIONS = ["Technical advice", "Product feedback", "Introductions", "Funding", "Design", "Marketing", "Legal"];
const URGENCY_OPTIONS = ["high", "medium", "low"];

// Mock profiles for matching
const MOCK_MATCHES = [
  {
    id: "match-1",
    name: "Maya Patel",
    role: "Principal Product Designer",
    company: "Figma",
    bio: "Leading design systems and mentoring designers globally.",
    skills: ["Figma", "Design Systems", "User Research"],
    domains: ["Developer Tools", "Enterprise SaaS"],
    score: 82,
    reason: "Strong alignment with your domain expertise and design collaboration needs",
  },
  {
    id: "match-2", 
    name: "Jordan Lee",
    role: "Senior ML Engineer",
    company: "Anthropic",
    bio: "Working on AI safety and alignment research.",
    skills: ["PyTorch", "RLHF", "Transformers"],
    domains: ["AI/ML"],
    score: 76,
    reason: "Complementary technical skills in AI/ML with mentorship experience",
  },
  {
    id: "match-3",
    name: "Chris Wong",
    role: "Founding Engineer",
    company: "Vercel",
    bio: "Building the future of web development infrastructure.",
    skills: ["TypeScript", "React", "Edge Computing"],
    domains: ["Developer Tools", "Enterprise SaaS"],
    score: 63,
    reason: "Relevant industry experience with strong technical background",
  },
  {
    id: "match-4",
    name: "Taylor Kim",
    role: "Partner",
    company: "Sequoia Capital",
    bio: "Investing in early-stage AI and infrastructure companies.",
    skills: ["Due Diligence", "Market Analysis", "Board Governance"],
    domains: ["AI/ML", "FinTech"],
    score: 58,
    reason: "Could provide valuable introductions and funding connections",
  },
];

const selectStyle = {
  padding: "9px 14px",
  border: "1px solid #e2e8f0",
  borderRadius: "8px",
  fontSize: "0.9rem",
  background: "#fff",
  minWidth: "160px",
  outline: "none",
};

export default function SmartQuery() {
  const navigate = useNavigate();
  const [query, setQuery] = useState({
    looking_for: LOOKING_FOR_OPTIONS[0],
    domain: DOMAIN_OPTIONS[0],
    urgency: "medium",
    help_type: HELP_OPTIONS[0],
  });
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);

  async function handleSearch() {
    setLoading(true);
    setSearched(true);
    await new Promise(resolve => setTimeout(resolve, 800));
    setResults(MOCK_MATCHES);
    setLoading(false);
  }

  function getScoreColor(score) {
    if (score >= 80) return "#10b981";
    if (score >= 70) return "#6366f1";
    if (score >= 60) return "#f59e0b";
    return "#64748b";
  }

  return (
    <div style={{ maxWidth: "760px", margin: "0 auto", padding: "40px 20px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "16px", marginBottom: "28px" }}>
        <button onClick={() => navigate("/")} style={backBtn}>← Home</button>
        <h1 style={{ fontSize: "1.8rem", fontWeight: 800 }}>Find my people</h1>
      </div>

      <div style={{
        background: "#fff",
        borderRadius: "16px",
        padding: "24px",
        boxShadow: "0 2px 12px rgba(0,0,0,0.06)",
        marginBottom: "28px",
      }}>
        <p style={{ fontWeight: 600, marginBottom: "14px", color: "#1e293b" }}>Find connections who can help you with…</p>

        <div style={{ display: "flex", flexWrap: "wrap", gap: "12px", alignItems: "flex-end" }}>
          <div>
            <label style={labelStyle}>Looking for</label>
            <select style={selectStyle} value={query.looking_for} onChange={(e) => setQuery((q) => ({ ...q, looking_for: e.target.value }))}>
              {LOOKING_FOR_OPTIONS.map((o) => <option key={o}>{o}</option>)}
            </select>
          </div>

          <div>
            <label style={labelStyle}>Domain</label>
            <select style={selectStyle} value={query.domain} onChange={(e) => setQuery((q) => ({ ...q, domain: e.target.value }))}>
              {DOMAIN_OPTIONS.map((o) => <option key={o}>{o}</option>)}
            </select>
          </div>

          <div>
            <label style={labelStyle}>Help type</label>
            <select style={selectStyle} value={query.help_type} onChange={(e) => setQuery((q) => ({ ...q, help_type: e.target.value }))}>
              {HELP_OPTIONS.map((o) => <option key={o}>{o}</option>)}
            </select>
          </div>

          <div>
            <label style={labelStyle}>Urgency</label>
            <select style={selectStyle} value={query.urgency} onChange={(e) => setQuery((q) => ({ ...q, urgency: e.target.value }))}>
              {URGENCY_OPTIONS.map((o) => <option key={o}>{o}</option>)}
            </select>
          </div>

          <button
            onClick={handleSearch}
            disabled={loading}
            style={{
              padding: "10px 26px",
              background: loading ? "#a5b4fc" : "linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)",
              color: "#fff",
              border: "none", borderRadius: "8px",
              fontWeight: 700, cursor: loading ? "default" : "pointer",
              alignSelf: "flex-end",
            }}
          >
            {loading ? "Searching…" : "Find Connections"}
          </button>
        </div>
      </div>

      {!searched && (
        <p style={{ color: "#94a3b8", textAlign: "center", marginTop: "40px" }}>
          Set your criteria above and click "Find Connections" to see AI-ranked matches.
        </p>
      )}

      {loading && (
        <div style={{ textAlign: "center", padding: "40px" }}>
          <div style={{
            width: "40px",
            height: "40px",
            border: "3px solid #e2e8f0",
            borderTopColor: "#6366f1",
            borderRadius: "50%",
            margin: "0 auto 16px",
            animation: "spin 0.8s linear infinite"
          }}></div>
          <p style={{ color: "#64748b" }}>Finding your people...</p>
          <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
        </div>
      )}

      {!loading && results.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          {results.map((match) => (
            <div key={match.id} style={{
              background: "#fff",
              borderRadius: "16px",
              padding: "20px 24px",
              boxShadow: "0 2px 12px rgba(0,0,0,0.06)",
              display: "flex",
              alignItems: "center",
              gap: "20px",
            }}>
              <div style={{
                width: "56px",
                height: "56px",
                borderRadius: "14px",
                background: "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
                color: "white",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: "22px",
                fontWeight: 700,
                flexShrink: 0,
              }}>
                {match.name.charAt(0)}
              </div>
              
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "4px" }}>
                  <h3 style={{ fontSize: "1.1rem", fontWeight: 700, color: "#1e293b" }}>
                    {match.name}
                  </h3>
                  <span style={{
                    background: `${getScoreColor(match.score)}15`,
                    color: getScoreColor(match.score),
                    padding: "3px 10px",
                    borderRadius: "999px",
                    fontSize: "0.8rem",
                    fontWeight: 700,
                  }}>
                    {match.score}% match
                  </span>
                </div>
                <p style={{ fontSize: "0.9rem", color: "#64748b", marginBottom: "6px" }}>
                  {match.role} @ {match.company}
                </p>
                <p style={{ fontSize: "0.85rem", color: "#94a3b8", fontStyle: "italic" }}>
                  "{match.reason}"
                </p>
              </div>

              <button 
                onClick={() => navigate(`/profile/${match.id}`)} 
                style={{
                  padding: "8px 18px",
                  background: "linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)",
                  color: "#fff",
                  border: "none",
                  borderRadius: "8px",
                  cursor: "pointer",
                  fontSize: "0.85rem",
                  fontWeight: 600,
                  flexShrink: 0,
                }}
              >
                View
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const backBtn = {
  padding: "6px 14px", background: "none",
  border: "1px solid #e2e8f0", borderRadius: "8px",
  cursor: "pointer", color: "#64748b", fontSize: "0.85rem",
};

const labelStyle = {
  display: "block",
  fontSize: "0.78rem",
  fontWeight: 600,
  color: "#94a3b8",
  textTransform: "uppercase",
  letterSpacing: "0.05em",
  marginBottom: "4px",
};
