import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import useUserId from "../hooks/useUserId.js";
import ProfileCard from "../components/ProfileCard.jsx";

// Mock network contacts
const MOCK_CONTACTS = [
  {
    id: "mock-1",
    name: "Siddhi Bansal",
    role: "Software Engineer",
    company: "Stripe",
    bio: "Building payments infrastructure. Interested in fintech and developer tools.",
    skills: ["Python", "Go", "Distributed Systems"],
    looking_for: ["Collaborators", "Co-founder"],
    can_help_with: ["Technical advice", "Introductions"],
    domains: ["FinTech", "Developer Tools"],
    linkedin_url: "https://linkedin.com/in/siddhi-bansal",
  },
  {
    id: "mock-2",
    name: "Marcus Chen",
    role: "Product Manager",
    company: "Notion",
    bio: "Passionate about productivity tools and collaborative software.",
    skills: ["Product Strategy", "User Research", "SQL"],
    looking_for: ["Mentorship", "Advice"],
    can_help_with: ["Product feedback", "Introductions"],
    domains: ["Enterprise SaaS", "Consumer"],
    linkedin_url: "https://linkedin.com/in/marcus-chen",
  },
  {
    id: "mock-3",
    name: "Priya Sharma",
    role: "ML Engineer",
    company: "OpenAI",
    bio: "Working on large language models and AI safety research.",
    skills: ["PyTorch", "Transformers", "RLHF"],
    looking_for: ["Collaborators"],
    can_help_with: ["Technical advice"],
    domains: ["AI/ML"],
    linkedin_url: "https://linkedin.com/in/priya-sharma",
  },
  {
    id: "mock-4",
    name: "Jordan Rivera",
    role: "Founding Engineer",
    company: "Stealth Startup",
    bio: "Ex-Google, building something new in the climate space.",
    skills: ["React", "Node.js", "AWS"],
    looking_for: ["Co-founder", "Investors"],
    can_help_with: ["Technical advice", "Product feedback"],
    domains: ["Climate", "Consumer"],
    linkedin_url: "https://linkedin.com/in/jordan-rivera",
  },
  {
    id: "mock-5",
    name: "Emily Zhang",
    role: "Design Lead",
    company: "Figma",
    bio: "Leading design systems. Love mentoring junior designers.",
    skills: ["Figma", "Design Systems", "Prototyping"],
    looking_for: ["Mentorship"],
    can_help_with: ["Design", "Product feedback"],
    domains: ["Developer Tools", "Enterprise SaaS"],
    linkedin_url: "https://linkedin.com/in/emily-zhang",
  },
  {
    id: "mock-6",
    name: "Alex Kim",
    role: "VC Associate",
    company: "a16z",
    bio: "Investing in early-stage AI and dev tools companies.",
    skills: ["Due Diligence", "Market Analysis", "Networking"],
    looking_for: ["Advice"],
    can_help_with: ["Funding", "Introductions"],
    domains: ["AI/ML", "Developer Tools"],
    linkedin_url: "https://linkedin.com/in/alex-kim",
  },
];

export default function MyNetwork() {
  const userId = useUserId();
  const navigate = useNavigate();
  const [contacts, setContacts] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Simulate loading
    setTimeout(() => {
      setContacts(MOCK_CONTACTS);
      setLoading(false);
    }, 500);
  }, [userId]);

  function handleRemove(profileId) {
    setContacts((prev) => prev.filter((c) => c.id !== profileId));
  }

  return (
    <div style={{ maxWidth: "960px", margin: "0 auto", padding: "40px 20px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "16px", marginBottom: "28px" }}>
        <button onClick={() => navigate("/")} style={backBtn}>← Home</button>
        <h1 style={{ fontSize: "1.8rem", fontWeight: 800 }}>My Network</h1>
        <span style={{ 
          background: "#e0e7ff", 
          color: "#4338ca", 
          padding: "4px 12px", 
          borderRadius: "999px", 
          fontSize: "0.85rem",
          fontWeight: 600
        }}>
          {contacts.length} contacts
        </span>
      </div>

      {loading && <p style={{ color: "#888" }}>Loading…</p>}

      {!loading && contacts.length === 0 && (
        <div style={{ textAlign: "center", padding: "60px 0", color: "#aaa" }}>
          <p style={{ fontSize: "1.1rem" }}>No contacts saved yet.</p>
          <p style={{ marginTop: "8px", fontSize: "0.9rem" }}>
            Scan someone's QR code or visit their profile to save them.
          </p>
        </div>
      )}

      <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
        {contacts.map((contact) => (
          <div key={contact.id} style={{
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
              {contact.name.charAt(0)}
            </div>
            
            <div style={{ flex: 1, minWidth: 0 }}>
              <h3 style={{ fontSize: "1.1rem", fontWeight: 700, color: "#1e293b", marginBottom: "4px" }}>
                {contact.name}
              </h3>
              <p style={{ fontSize: "0.9rem", color: "#64748b" }}>
                {contact.role} @ {contact.company}
              </p>
              <div style={{ display: "flex", gap: "6px", marginTop: "8px", flexWrap: "wrap" }}>
                {contact.domains?.slice(0, 3).map((d) => (
                  <span key={d} style={{
                    background: "#f1f5f9",
                    color: "#475569",
                    padding: "3px 10px",
                    borderRadius: "999px",
                    fontSize: "0.75rem",
                    fontWeight: 500,
                  }}>{d}</span>
                ))}
              </div>
            </div>

            <div style={{ display: "flex", gap: "8px", flexShrink: 0 }}>
              <button onClick={() => navigate(`/profile/${contact.id}`)} style={viewBtn}>
                View
              </button>
              <button onClick={() => handleRemove(contact.id)} style={removeBtn}>
                Remove
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

const backBtn = {
  padding: "6px 14px", background: "none",
  border: "1px solid #ddd", borderRadius: "8px",
  cursor: "pointer", color: "#555", fontSize: "0.85rem",
};
const viewBtn = {
  padding: "8px 18px",
  background: "linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)", 
  color: "#fff",
  border: "none", borderRadius: "8px", cursor: "pointer",
  fontSize: "0.85rem", fontWeight: 600,
};
const removeBtn = {
  padding: "8px 14px",
  background: "#fef2f2", color: "#dc2626",
  border: "none", borderRadius: "8px", cursor: "pointer",
  fontSize: "0.85rem", fontWeight: 500,
};
