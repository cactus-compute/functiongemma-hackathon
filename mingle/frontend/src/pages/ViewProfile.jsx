import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { getProfile, getQR, saveContact } from "../api/client.js";
import useUserId from "../hooks/useUserId.js";
import ProfileCard from "../components/ProfileCard.jsx";
import QRCodeModal from "../components/QRCodeModal.jsx";

export default function ViewProfile() {
  const { id } = useParams();
  const userId = useUserId();
  const [profile, setProfile] = useState(null);
  const [qrData, setQrData] = useState(null);
  const [showQR, setShowQR] = useState(false);
  const [saved, setSaved] = useState(false);
  const [toast, setToast] = useState(null);
  const [loading, setLoading] = useState(true);

  const isMyCard = localStorage.getItem("mingle_my_profile_id") === id;

  useEffect(() => {
    getProfile(id)
      .then(setProfile)
      .catch(() => setProfile(null))
      .finally(() => setLoading(false));
    getQR(id)
      .then(setQrData)
      .catch(() => {});
  }, [id]);

  async function handleSave() {
    try {
      await saveContact(userId, id);
      setSaved(true);
      showToast("Saved to your network!");
    } catch (err) {
      showToast(err.response?.data?.error || "Could not save contact");
    }
  }

  function showToast(msg) {
    setToast(msg);
    setTimeout(() => setToast(null), 3000);
  }

  if (loading) return <CenteredMsg>Loading…</CenteredMsg>;
  if (!profile) return <CenteredMsg>Profile not found.</CenteredMsg>;

  const actions = (
    <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
      {isMyCard ? (
        <button
          onClick={() => setShowQR(true)}
          style={btnStyle("#6c63ff")}
        >
          Share my card (QR)
        </button>
      ) : (
        <button
          onClick={handleSave}
          disabled={saved}
          style={btnStyle(saved ? "#aaa" : "#2ecc71")}
        >
          {saved ? "Saved!" : "Save to My Network"}
        </button>
      )}
    </div>
  );

  return (
    <div style={{
      minHeight: "100vh",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      padding: "60px 20px",
      background: "linear-gradient(135deg, #f8f9fa 0%, #e8f4ff 100%)",
    }}>
      <ProfileCard profile={profile} actions={actions} />

      {showQR && qrData && (
        <QRCodeModal qr={qrData.qr} url={qrData.url} onClose={() => setShowQR(false)} />
      )}

      {toast && (
        <div style={{
          position: "fixed", bottom: "32px", left: "50%", transform: "translateX(-50%)",
          background: "#1a1a2e", color: "#fff",
          padding: "10px 24px", borderRadius: "999px",
          fontSize: "0.9rem", pointerEvents: "none",
          animation: "fadein 0.2s ease",
        }}>
          {toast}
        </div>
      )}
    </div>
  );
}

function CenteredMsg({ children }) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100vh" }}>
      <p style={{ color: "#888" }}>{children}</p>
    </div>
  );
}

function btnStyle(bg) {
  return {
    padding: "10px 22px",
    background: bg,
    color: "#fff",
    border: "none",
    borderRadius: "8px",
    cursor: "pointer",
    fontWeight: 600,
    fontSize: "0.9rem",
  };
}
