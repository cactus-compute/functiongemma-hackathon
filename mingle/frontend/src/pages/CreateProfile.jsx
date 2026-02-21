import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { createProfile } from "../api/client.js";

const LOOKING_FOR_OPTIONS = ["Co-founder", "Collaborators", "Mentorship", "Investors", "Customers", "Employees", "Advice"];
const HELP_OPTIONS = ["Technical advice", "Product feedback", "Introductions", "Funding", "Design", "Marketing", "Legal"];
const DOMAIN_OPTIONS = ["AI/ML", "Hardware", "FinTech", "HealthTech", "EdTech", "Climate", "Web3", "Enterprise SaaS", "Consumer", "Developer Tools"];

const inputStyle = {
  width: "100%",
  padding: "10px 14px",
  border: "1px solid #ddd",
  borderRadius: "8px",
  fontSize: "0.95rem",
  outline: "none",
  marginBottom: "0",
};

const fieldStyle = { marginBottom: "18px" };
const labelStyle = { display: "block", fontWeight: 600, marginBottom: "6px", fontSize: "0.9rem" };

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
    <div style={{ border: "1px solid #ddd", borderRadius: "8px", padding: "6px 10px", display: "flex", flexWrap: "wrap", gap: "4px" }}>
      {value.map((chip, i) => (
        <span key={i} style={{
          background: "#e8f4ff", color: "#0066cc",
          borderRadius: "999px", padding: "2px 10px",
          fontSize: "0.82rem", display: "flex", alignItems: "center", gap: "4px"
        }}>
          {chip}
          <button onClick={() => removeChip(i)} style={{ background: "none", border: "none", cursor: "pointer", color: "#0066cc", padding: 0, lineHeight: 1 }}>×</button>
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
          padding: "5px 12px",
          border: `1px solid ${value.includes(opt) ? "#6c63ff" : "#ddd"}`,
          borderRadius: "999px",
          cursor: "pointer",
          background: value.includes(opt) ? "#f0eeff" : "#fff",
          fontSize: "0.85rem",
          userSelect: "none",
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

  function set(field) {
    return (val) => setForm((prev) => ({ ...prev, [field]: val }));
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await createProfile(form);
      localStorage.setItem("mingle_my_profile_id", res.id);
      navigate(`/profile/${res.id}`);
    } catch (err) {
      setError(err.response?.data?.error || err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ maxWidth: "600px", margin: "0 auto", padding: "40px 20px" }}>
      <h1 style={{ fontSize: "1.8rem", fontWeight: 800, marginBottom: "8px" }}>Create your profile</h1>
      <p style={{ color: "#666", marginBottom: "28px" }}>Your digital business card for smarter networking.</p>

      <form onSubmit={handleSubmit}>
        {[["name", "Full Name"], ["role", "Role / Title"], ["company", "Company"]].map(([field, placeholder]) => (
          <div key={field} style={fieldStyle}>
            <label style={labelStyle}>{placeholder}</label>
            <input
              style={inputStyle}
              value={form[field]}
              onChange={(e) => set(field)(e.target.value)}
              placeholder={placeholder}
              required
            />
          </div>
        ))}

        <div style={fieldStyle}>
          <label style={labelStyle}>Bio</label>
          <textarea
            style={{ ...inputStyle, resize: "vertical" }}
            value={form.bio}
            onChange={(e) => set("bio")(e.target.value)}
            placeholder="A brief description of what you're working on…"
            rows={3}
            required
          />
        </div>

        <div style={fieldStyle}>
          <label style={labelStyle}>Skills <span style={{ fontWeight: 400, color: "#888" }}>(press Enter to add)</span></label>
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
          <label style={labelStyle}>LinkedIn URL <span style={{ fontWeight: 400, color: "#888" }}>(optional)</span></label>
          <input
            style={inputStyle}
            value={form.linkedin_url}
            onChange={(e) => set("linkedin_url")(e.target.value)}
            placeholder="https://linkedin.com/in/your-profile"
            type="url"
          />
        </div>

        {error && <p style={{ color: "red", marginBottom: "12px" }}>{error}</p>}

        <button
          type="submit"
          disabled={loading}
          style={{
            width: "100%", padding: "13px",
            background: "#6c63ff", color: "#fff",
            border: "none", borderRadius: "10px",
            fontWeight: 700, fontSize: "1rem", cursor: "pointer",
            opacity: loading ? 0.7 : 1,
          }}
        >
          {loading ? "Creating…" : "Create Profile"}
        </button>
      </form>
    </div>
  );
}
