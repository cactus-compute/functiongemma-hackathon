export default function QRCodeModal({ qr, url, onClose }) {
  if (!qr) return null;

  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed", inset: 0,
        background: "rgba(0,0,0,0.5)",
        display: "flex", alignItems: "center", justifyContent: "center",
        zIndex: 1000,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: "#fff",
          borderRadius: "16px",
          padding: "32px",
          textAlign: "center",
          maxWidth: "360px",
          width: "90%",
        }}
      >
        <h3 style={{ marginBottom: "16px", fontSize: "1.1rem" }}>Share your card</h3>
        <img src={qr} alt="QR code" style={{ width: "100%", maxWidth: "280px" }} />
        {url && (
          <p style={{ marginTop: "12px", fontSize: "0.8rem", color: "#666", wordBreak: "break-all" }}>
            {url}
          </p>
        )}
        <button
          onClick={onClose}
          style={{
            marginTop: "20px", padding: "8px 24px",
            background: "#1a1a2e", color: "#fff",
            border: "none", borderRadius: "8px", cursor: "pointer",
          }}
        >
          Close
        </button>
      </div>
    </div>
  );
}
