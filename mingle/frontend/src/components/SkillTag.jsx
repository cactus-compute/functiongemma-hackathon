export default function SkillTag({ label }) {
  return (
    <span style={{
      display: "inline-block",
      background: "#e8f4ff",
      color: "#0066cc",
      borderRadius: "999px",
      padding: "2px 10px",
      fontSize: "0.78rem",
      fontWeight: 500,
      margin: "2px",
    }}>
      {label}
    </span>
  );
}
