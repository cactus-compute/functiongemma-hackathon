import { useState } from "react";
import SkillTag from "./SkillTag.jsx";
import OutreachModal from "./OutreachModal.jsx";

export default function MatchCard({ ranking, profile, senderProfile }) {
  const [showOutreach, setShowOutreach] = useState(false);
  const score = Math.round((ranking.match_score || 0) * 100);

  return (
    <div style={{
      background: "#fff",
      borderRadius: "12px",
      padding: "20px",
      boxShadow: "0 2px 8px rgba(0,0,0,0.07)",
      marginBottom: "16px",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <h3 style={{ fontSize: "1.05rem", fontWeight: 700 }}>{profile?.name}</h3>
          <p style={{ fontSize: "0.85rem", color: "#666" }}>
            {profile?.role} at {profile?.company}
          </p>
        </div>
        <div style={{
          background: score >= 70 ? "#d4f7dc" : score >= 40 ? "#fff3cd" : "#fde8e8",
          color: score >= 70 ? "#1a7a3c" : score >= 40 ? "#856404" : "#c0392b",
          borderRadius: "999px",
          padding: "4px 12px",
          fontWeight: 700,
          fontSize: "0.9rem",
          whiteSpace: "nowrap",
        }}>
          {score}% match
        </div>
      </div>

      {/* Score bar */}
      <div style={{ margin: "12px 0 8px", height: "6px", background: "#eee", borderRadius: "999px", overflow: "hidden" }}>
        <div style={{
          height: "100%",
          width: `${score}%`,
          background: score >= 70 ? "#2ecc71" : score >= 40 ? "#f39c12" : "#e74c3c",
          borderRadius: "999px",
          transition: "width 0.4s ease",
        }} />
      </div>

      {ranking.match_reason && (
        <p style={{ fontSize: "0.85rem", color: "#444", marginBottom: "8px" }}>
          {ranking.match_reason}
        </p>
      )}

      {ranking.outreach_angle && (
        <p style={{ fontSize: "0.82rem", color: "#888", fontStyle: "italic", marginBottom: "10px" }}>
          Suggested angle: {ranking.outreach_angle}
        </p>
      )}

      {profile?.skills?.length > 0 && (
        <div style={{ marginBottom: "10px" }}>
          {profile.skills.slice(0, 5).map((s) => <SkillTag key={s} label={s} />)}
        </div>
      )}

      {ranking.source && (
        <p style={{ fontSize: "0.72rem", color: "#bbb", marginBottom: "8px" }}>
          Ranked via: {ranking.source}
        </p>
      )}

      {senderProfile && (
        <button
          onClick={() => setShowOutreach(true)}
          style={{
            padding: "7px 18px",
            background: "#6c63ff", color: "#fff",
            border: "none", borderRadius: "8px", cursor: "pointer",
            fontSize: "0.85rem",
          }}
        >
          Draft Message
        </button>
      )}

      {showOutreach && (
        <OutreachModal
          sender={senderProfile}
          recipient={profile}
          context={ranking.outreach_angle}
          onClose={() => setShowOutreach(false)}
        />
      )}
    </div>
  );
}
