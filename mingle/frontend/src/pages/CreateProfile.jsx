import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { createProfile } from "../api/client.js";

const LOOKING_FOR_OPTIONS = ["Co-founder", "Collaborators", "Mentorship", "Investors", "Customers", "Employees", "Advice"];
const HELP_OPTIONS = ["Technical advice", "Product feedback", "Introductions", "Funding", "Design", "Marketing", "Legal"];
const DOMAIN_OPTIONS = ["AI/ML", "Hardware", "FinTech", "HealthTech", "EdTech", "Climate", "Web3", "Enterprise SaaS", "Consumer", "Developer Tools"];

const inputStyle = {
  width: "100%",
  padding: "12px 16px",
  border: "1px solid #e2e8f0",
  borderRadius: "10px",
  fontSize: "0.95rem",
  outline: "none",
  transition: "border-color 0.2s, box-shadow 0.2s",
};

const fieldStyle = { marginBottom: "20px" };
const labelStyle = { display: "block", fontWeight: 600, marginBottom: "8px", fontSize: "0.9rem", color: "#374151" };

function ChipInput({ value, onChange, placeholder }) {
  const [input, setInput] = useState("");

  function addChip(e) {
    if ((e.key === "Enter" || e.key === ",") && input.trim()) {
      e.preventDefault();
      const newVal = [...value, input.trim()];
      onChange(newVal);
      setInput("");
    }
  }

  function removeChip(idx) {
    onChange(value.filter((_, i) => i !== idx));
  }

  return (
    <div style={{ border: "1px solid #e2e8f0", borderRadius: "10px", padding: "8px 12px", display: "flex", flexWrap: "wrap", gap: "6px", background: "#fff" }}>
      {value.map((chip, i) => (
        <span key={i} style={{
          background: "linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%)", 
          color: "#4338ca",
          borderRadius: "999px", padding: "4px 12px",
          fontSize: "0.82rem", display: "flex", alignItems: "center", gap: "6px",
          fontWeight: 500
        }}>
          {chip}
          <button onClick={() => removeChip(i)} style={{ background: "none", border: "none", cursor: "pointer", color: "#4338ca", padding: 0, lineHeight: 1, fontSize: "16px" }}>×</button>
        </span>
      ))}
      <input
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={addChip}
        placeholder={value.length === 0 ? placeholder : "Add more…"}
        style={{ border: "none", outline: "none", fontSize: "0.9rem", minWidth: "120px", flex: 1 }}
      />
    </div>
  );
}

function CheckboxGroup({ options, value, onChange }) {
  function toggle(opt) {
    if (value.includes(opt)) {
      onChange(value.filter((v) => v !== opt));
    } else {
      onChange([...value, opt]);
    }
  }
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
      {options.map((opt) => (
        <label key={opt} style={{
          display: "flex", alignItems: "center", gap: "5px",
          padding: "6px 14px",
          border: `1px solid ${value.includes(opt) ? "#6366f1" : "#e2e8f0"}`,
          borderRadius: "999px",
          cursor: "pointer",
          background: value.includes(opt) ? "linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%)" : "#fff",
          fontSize: "0.85rem",
          userSelect: "none",
          color: value.includes(opt) ? "#4338ca" : "#64748b",
          fontWeight: value.includes(opt) ? 600 : 400,
          transition: "all 0.15s ease",
        }}>
          <input
            type="checkbox"
            checked={value.includes(opt)}
            onChange={() => toggle(opt)}
            style={{ display: "none" }}
          />
          {opt}
        </label>
      ))}
    </div>
  );
}

export default function CreateProfile() {
  const navigate = useNavigate();
  const [form, setForm] = useState({
    name: "", role: "", company: "", bio: "",
    skills: [], looking_for: [], can_help_with: [], domains: [],
    linkedin_url: "",
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  const [createdId, setCreatedId] = useState(null);

  function set(field) {
    return (val) => setForm((prev) => ({ ...prev, [field]: val }));
  }

  // Generate LinkedIn placeholder based on name
  const linkedinPlaceholder = form.name 
    ? `https://linkedin.com/in/${form.name.toLowerCase().replace(/\s+/g, "-").replace(/[^a-z0-9-]/g, "")}`
    : "https://linkedin.com/in/your-profile";

  async function handleSubmit(e) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await createProfile(form);
      localStorage.setItem("mingle_my_profile_id", res.id);
      setCreatedId(res.id);
      setSuccess(true);
      // Navigate after showing success
      setTimeout(() => {
        navigate(`/profile/${res.id}`);
      }, 1500);
    } catch (err) {
      setError(err.response?.data?.error || err.message);
    } finally {
      setLoading(false);
    }
  }

  // Success State
  if (success) {
    return (
      <div style={{ 
        minHeight: "80vh", 
        display: "flex", 
        flexDirection: "column",
        alignItems: "center", 
        justifyContent: "center",
        padding: "40px 20px",
        textAlign: "center"
      }}>
        <div className="fade-in" style={{
          width: "100px",
          height: "100px",
          borderRadius: "50%",
          background: "linear-gradient(135deg, #10b981 0%, #059669 100%)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          marginBottom: "24px",
          boxShadow: "0 10px 40px rgba(16, 185, 129, 0.3)",
        }}>
          <span style={{ fontSize: "48px" }}>✓</span>
        </div>
        <h2 style={{ fontSize: "1.8rem", fontWeight: 700, color: "#1e293b", marginBottom: "12px" }}>
          Profile Created!
        </h2>
        <p style={{ color: "#64748b", marginBottom: "8px" }}>
          Welcome to Mingle, {form.name.split(" ")[0]}! 🎉
        </p>
        <p style={{ color: "#94a3b8", fontSize: "14px" }}>
          Redirecting to your profile...
        </p>
      </div>
    );
  }

  return (
    <div style={{ maxWidth: "640px", margin: "0 auto", padding: "40px 20px" }}>
      {/* Header */}
      <div style={{ marginBottom: "32px" }}>
        <button 
          onClick={() => navigate("/")} 
          style={{
            padding: "8px 16px",
            background: "#f1f5f9",
            border: "none",
            borderRadius: "8px",
            cursor: "pointer",
            color: "#475569",
            fontSize: "14px",
            fontWeight: 500,
            marginBottom: "20px",
          }}
        >
          ← Home
        </button>
        <h1 style={{ fontSize: "2rem", fontWeight: 800, color: "#1e293b", marginBottom: "8px" }}>
          Create your profile
        </h1>
        <p style={{ color: "#64748b", fontSize: "15px" }}>
          Your digital business card for smarter networking.
        </p>
      </div>

      <form onSubmit={handleSubmit} style={{
        background: "#fff",
        borderRadius: "20px",
        padding: "32px",
        boxShadow: "0 4px 20px rgba(0,0,0,0.06)",
      }}>
        {[["name", "Full Name"], ["role", "Role / Title"], ["company", "Company"]].map(([field, placeholder]) => (
          <div key={field} style={fieldStyle}>
            <label style={labelStyle}>{placeholder}</label>
            <input
              style={inputStyle}
              value={form[field]}
              onChange={(e) => set(field)(e.target.value)}
              placeholder={placeholder}
              
            />
          </div>
        ))}

        <div style={fieldStyle}>
          <label style={labelStyle}>Bio</label>
          <textarea
            style={{ ...inputStyle, resize: "vertical", minHeight: "80px" }}
            value={form.bio}
            onChange={(e) => set("bio")(e.target.value)}
            placeholder="A brief description of what you're working on…"
            rows={3}
            
          />
        </div>

        <div style={fieldStyle}>
          <label style={labelStyle}>
            Skills 
            <span style={{ fontWeight: 400, color: "#94a3b8", marginLeft: "8px" }}>(press Enter to add)</span>
          </label>
          <ChipInput value={form.skills} onChange={set("skills")} placeholder="Python, Product Design…" />
        </div>

        <div style={fieldStyle}>
          <label style={labelStyle}>Looking for</label>
          <CheckboxGroup options={LOOKING_FOR_OPTIONS} value={form.looking_for} onChange={set("looking_for")} />
        </div>

        <div style={fieldStyle}>
          <label style={labelStyle}>Can help with</label>
          <CheckboxGroup options={HELP_OPTIONS} value={form.can_help_with} onChange={set("can_help_with")} />
        </div>

        <div style={fieldStyle}>
          <label style={labelStyle}>Domains</label>
          <CheckboxGroup options={DOMAIN_OPTIONS} value={form.domains} onChange={set("domains")} />
        </div>

        <div style={fieldStyle}>
          <label style={labelStyle}>
            LinkedIn URL 
            <span style={{ fontWeight: 400, color: "#94a3b8", marginLeft: "8px" }}>(optional)</span>
          </label>
          <input
            style={inputStyle}
            value={form.linkedin_url}
            onChange={(e) => set("linkedin_url")(e.target.value)}
            placeholder={linkedinPlaceholder}
            type="url"
          />
        </div>

        {error && (
          <div style={{ 
            padding: "12px 16px", 
            background: "#fef2f2", 
            borderRadius: "10px", 
            color: "#dc2626", 
            marginBottom: "16px",
            fontSize: "14px"
          }}>
            ⚠️ {error}
          </div>
        )}

        <button
          type="submit"
          disabled={loading}
          style={{
            width: "100%", 
            padding: "14px",
            background: loading ? "#94a3b8" : "linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)",
            color: "#fff",
            border: "none", 
            borderRadius: "12px",
            fontWeight: 600, 
            fontSize: "1rem", 
            cursor: loading ? "not-allowed" : "pointer",
            transition: "all 0.2s",
            boxShadow: loading ? "none" : "0 4px 15px rgba(99, 102, 241, 0.3)",
          }}
        >
          {loading ? "Creating..." : "Create Profile"}
        </button>
      </form>
    </div>
  );
}
