import SkillTag from "./SkillTag.jsx";

const cardStyle = {
  background: "#fff",
  borderRadius: "16px",
  padding: "24px",
  boxShadow: "0 2px 12px rgba(0,0,0,0.08)",
  maxWidth: "480px",
  width: "100%",
};

export default function ProfileCard({ profile, actions }) {
  if (!profile) return null;
  return (
    <div style={cardStyle}>
      <div style={{ marginBottom: "12px" }}>
        <h2 style={{ fontSize: "1.4rem", fontWeight: 700 }}>{profile.name}</h2>
        <p style={{ color: "#555", fontSize: "0.95rem" }}>
          {profile.role} at <strong>{profile.company}</strong>
        </p>
      </div>

      {profile.bio && (
        <p style={{ color: "#333", marginBottom: "12px", lineHeight: 1.5 }}>{profile.bio}</p>
      )}

      {profile.skills?.length > 0 && (
        <div style={{ marginBottom: "10px" }}>
          <Label>Skills</Label>
          <div>{profile.skills.map((s) => <SkillTag key={s} label={s} />)}</div>
        </div>
      )}

      {profile.looking_for?.length > 0 && (
        <div style={{ marginBottom: "10px" }}>
          <Label>Looking for</Label>
          <div>{profile.looking_for.map((s) => <SkillTag key={s} label={s} />)}</div>
        </div>
      )}

      {profile.can_help_with?.length > 0 && (
        <div style={{ marginBottom: "10px" }}>
          <Label>Can help with</Label>
          <div>{profile.can_help_with.map((s) => <SkillTag key={s} label={s} />)}</div>
        </div>
      )}

      {profile.domains?.length > 0 && (
        <div style={{ marginBottom: "10px" }}>
          <Label>Domains</Label>
          <div>{profile.domains.map((s) => <SkillTag key={s} label={s} />)}</div>
        </div>
      )}

      {profile.linkedin_url && (
        <a
          href={profile.linkedin_url}
          target="_blank"
          rel="noopener noreferrer"
          style={{ color: "#0077b5", fontSize: "0.85rem", display: "block", marginTop: "8px" }}
        >
          LinkedIn Profile
        </a>
      )}

      {actions && <div style={{ marginTop: "16px" }}>{actions}</div>}
    </div>
  );
}

function Label({ children }) {
  return (
    <span style={{ fontSize: "0.75rem", fontWeight: 600, color: "#888", textTransform: "uppercase", letterSpacing: "0.05em" }}>
      {children}
    </span>
  );
}
