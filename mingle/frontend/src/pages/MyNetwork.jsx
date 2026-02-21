import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getNetwork, removeContact } from "../api/client.js";
import useUserId from "../hooks/useUserId.js";
import ProfileCard from "../components/ProfileCard.jsx";

export default function MyNetwork() {
  const userId = useUserId();
  const navigate = useNavigate();
  const [contacts, setContacts] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getNetwork(userId)
      .then(setContacts)
      .catch(() => setContacts([]))
      .finally(() => setLoading(false));
  }, [userId]);

  async function handleRemove(profileId) {
    await removeContact(userId, profileId);
    setContacts((prev) => prev.filter((c) => c.id !== profileId));
  }

  return (
    <div style={{ maxWidth: "960px", margin: "0 auto", padding: "40px 20px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "16px", marginBottom: "28px" }}>
        <button onClick={() => navigate("/")} style={backBtn}>← Home</button>
        <h1 style={{ fontSize: "1.8rem", fontWeight: 800 }}>My Network</h1>
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

      <div style={{ display: "flex", flexWrap: "wrap", gap: "20px" }}>
        {contacts.map((contact) => (
          <ProfileCard
            key={contact.id}
            profile={contact}
            actions={
              <div style={{ display: "flex", gap: "8px" }}>
                <button onClick={() => navigate(`/profile/${contact.id}`)} style={viewBtn}>
                  View Profile
                </button>
                <button onClick={() => handleRemove(contact.id)} style={removeBtn}>
                  Remove
                </button>
              </div>
            }
          />
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
  padding: "7px 16px",
  background: "#6c63ff", color: "#fff",
  border: "none", borderRadius: "8px", cursor: "pointer",
  fontSize: "0.85rem",
};
const removeBtn = {
  padding: "7px 16px",
  background: "#fde8e8", color: "#c0392b",
  border: "none", borderRadius: "8px", cursor: "pointer",
  fontSize: "0.85rem",
};
