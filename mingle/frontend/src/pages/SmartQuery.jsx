import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { getAllProfiles, getProfile, rankContacts } from "../api/client.js";
import useUserId from "../hooks/useUserId.js";
import MatchCard from "../components/MatchCard.jsx";

const LOOKING_FOR_OPTIONS = ["Co-founder", "Collaborators", "Mentorship", "Investors", "Customers", "Employees", "Advice"];
const DOMAIN_OPTIONS = ["AI/ML", "Hardware", "FinTech", "HealthTech", "EdTech", "Climate", "Web3", "Enterprise SaaS", "Consumer", "Developer Tools"];
const HELP_OPTIONS = ["Technical advice", "Product feedback", "Introductions", "Funding", "Design", "Marketing", "Legal"];
const URGENCY_OPTIONS = ["high", "medium", "low"];

const selectStyle = {
  padding: "9px 14px",
  border: "1px solid #ddd",
  borderRadius: "8px",
  fontSize: "0.9rem",
  background: "#fff",
  minWidth: "160px",
  outline: "none",
};

export default function SmartQuery() {
  const navigate = useNavigate();
  const userId = useUserId();
  const [allProfiles, setAllProfiles] = useState([]);
  const [senderProfile, setSenderProfile] = useState(null);
  const [query, setQuery] = useState({
    looking_for: LOOKING_FOR_OPTIONS[0],
    domain: DOMAIN_OPTIONS[0],
    urgency: "medium",
    help_type: HELP_OPTIONS[0],
  });
  const [rankings, setRankings] = useState([]);
  const [profileMap, setProfileMap] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    getAllProfiles().then(setAllProfiles).catch(() => {});
    const myId = localStorage.getItem("mingle_my_profile_id");
    if (myId) {
      getProfile(myId).then(setSenderProfile).catch(() => {});
    }
  }, []);

  async function handleSearch() {
    if (allProfiles.length === 0) {
      setError("No profiles found. Create some profiles first.");
      return;
    }
    setLoading(true);
    setError(null);
    setRankings([]);
    try {
      const myId = localStorage.getItem("mingle_my_profile_id");
      const candidates = allProfiles.filter((p) => p.id !== myId);
      if (candidates.length === 0) {
        setError("No other profiles to match against.");
        setLoading(false);
        return;
      }
      const map = {};
      candidates.forEach((p) => { map[p.id] = p; });
      setProfileMap(map);

      const res = await rankContacts({
        query_looking_for: query.looking_for,
        query_domain: query.domain,
        query_help_type: query.help_type,
        urgency: query.urgency,
        candidates,
      });
      setRankings(res.rankings || []);
    } catch (err) {
      setError(err.response?.data?.error || err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ maxWidth: "760px", margin: "0 auto", padding: "40px 20px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "16px", marginBottom: "28px" }}>
        <button onClick={() => navigate("/")} style={backBtn}>← Home</button>
        <h1 style={{ fontSize: "1.8rem", fontWeight: 800 }}>Find my people</h1>
      </div>

      <div style={{
        background: "#fff",
        borderRadius: "14px",
        padding: "24px",
        boxShadow: "0 2px 10px rgba(0,0,0,0.07)",
        marginBottom: "28px",
      }}>
        <p style={{ fontWeight: 600, marginBottom: "14px", color: "#444" }}>Find connections who can help you with…</p>

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
              background: "#6c63ff", color: "#fff",
              border: "none", borderRadius: "8px",
              fontWeight: 700, cursor: "pointer",
              opacity: loading ? 0.7 : 1,
              alignSelf: "flex-end",
            }}
          >
            {loading ? "Searching…" : "Find Connections"}
          </button>
        </div>
      </div>

      {error && (
        <div style={{ padding: "12px 16px", background: "#fde8e8", borderRadius: "8px", color: "#c0392b", marginBottom: "16px" }}>
          {error}
        </div>
      )}

      {!loading && rankings.length === 0 && !error && (
        <p style={{ color: "#aaa", textAlign: "center", marginTop: "40px" }}>
          Set your criteria above and click "Find Connections" to see AI-ranked matches.
        </p>
      )}

      {rankings.map((r) => (
        <MatchCard
          key={r.contact_id}
          ranking={r}
          profile={profileMap[r.contact_id]}
          senderProfile={senderProfile}
        />
      ))}
    </div>
  );
}

const backBtn = {
  padding: "6px 14px", background: "none",
  border: "1px solid #ddd", borderRadius: "8px",
  cursor: "pointer", color: "#555", fontSize: "0.85rem",
};

const labelStyle = {
  display: "block",
  fontSize: "0.78rem",
  fontWeight: 600,
  color: "#888",
  textTransform: "uppercase",
  letterSpacing: "0.05em",
  marginBottom: "4px",
};
