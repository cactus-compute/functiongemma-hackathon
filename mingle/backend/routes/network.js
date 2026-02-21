const express = require("express");
const db = require("../db");

const router = express.Router();

const ARRAY_FIELDS = ["skills", "looking_for", "can_help_with", "domains"];

function serialize(profile) {
  const out = { ...profile };
  ARRAY_FIELDS.forEach((f) => {
    try {
      out[f] = JSON.parse(out[f]);
    } catch {
      out[f] = [];
    }
  });
  return out;
}

// GET /api/network?userId=x
router.get("/", (req, res) => {
  const { userId } = req.query;
  if (!userId) return res.status(400).json({ error: "userId query param required" });

  const rows = db
    .prepare(`
      SELECT p.*, n.saved_at
      FROM network n
      JOIN profiles p ON p.id = n.profile_id
      WHERE n.owner_user_id = ?
      ORDER BY n.saved_at DESC
    `)
    .all(userId);

  res.json(rows.map(serialize));
});

// POST /api/network
router.post("/", (req, res) => {
  const { userId, profileId } = req.body;
  if (!userId || !profileId) {
    return res.status(400).json({ error: "userId and profileId are required" });
  }

  const profile = db.prepare("SELECT id FROM profiles WHERE id = ?").get(profileId);
  if (!profile) return res.status(404).json({ error: "Profile not found" });

  try {
    db.prepare(
      "INSERT OR IGNORE INTO network (owner_user_id, profile_id) VALUES (?, ?)"
    ).run(userId, profileId);
    res.status(201).json({ status: "saved" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// DELETE /api/network/:profileId?userId=x
router.delete("/:profileId", (req, res) => {
  const { userId } = req.query;
  if (!userId) return res.status(400).json({ error: "userId query param required" });

  db.prepare(
    "DELETE FROM network WHERE owner_user_id = ? AND profile_id = ?"
  ).run(userId, req.params.profileId);

  res.json({ status: "removed" });
});

module.exports = router;
