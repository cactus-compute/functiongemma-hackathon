const express = require("express");
const { v4: uuidv4 } = require("uuid");
const axios = require("axios");
const db = require("../db");

const router = express.Router();
const AI_SERVER = process.env.AI_SERVER_URL || "http://localhost:8001";

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

function syncRag(profile) {
  axios
    .post(`${AI_SERVER}/ai/sync-profile-rag`, {
      profile_id: profile.id,
      name: profile.name,
      role: profile.role,
      company: profile.company,
      bio: profile.bio,
      skills: profile.skills || [],
      looking_for: profile.looking_for || [],
      can_help_with: profile.can_help_with || [],
      domains: profile.domains || [],
      linkedin_url: profile.linkedin_url || "",
    })
    .catch((err) => {
      console.warn("RAG sync failed (non-fatal):", err.message);
    });
}

// POST /api/profiles
router.post("/", (req, res) => {
  const { name, role, company, bio, skills, looking_for, can_help_with, domains, linkedin_url } =
    req.body;
  }
  const id = uuidv4();
  const stmt = db.prepare(`
    INSERT INTO profiles (id, name, role, company, bio, skills, looking_for, can_help_with, domains, linkedin_url)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);
  stmt.run(
    id,
    name,
    role,
    company,
    bio,
    JSON.stringify(skills || []),
    JSON.stringify(looking_for || []),
    JSON.stringify(can_help_with || []),
    JSON.stringify(domains || []),
    linkedin_url || ""
  );
  const profile = serialize(db.prepare("SELECT * FROM profiles WHERE id = ?").get(id));
  syncRag(profile);
  res.status(201).json({ id, profile });
});

// GET /api/profiles
router.get("/", (_req, res) => {
  const rows = db.prepare("SELECT * FROM profiles ORDER BY created_at DESC").all();
  res.json(rows.map(serialize));
});

// GET /api/profiles/:id
router.get("/:id", (req, res) => {
  const row = db.prepare("SELECT * FROM profiles WHERE id = ?").get(req.params.id);
  if (!row) return res.status(404).json({ error: "Profile not found" });
  res.json(serialize(row));
});

// PUT /api/profiles/:id
router.put("/:id", (req, res) => {
  const existing = db.prepare("SELECT * FROM profiles WHERE id = ?").get(req.params.id);
  if (!existing) return res.status(404).json({ error: "Profile not found" });

  const { name, role, company, bio, skills, looking_for, can_help_with, domains, linkedin_url } =
    req.body;

  db.prepare(`
    UPDATE profiles SET
      name = ?, role = ?, company = ?, bio = ?,
      skills = ?, looking_for = ?, can_help_with = ?, domains = ?, linkedin_url = ?
    WHERE id = ?
  `).run(
    name || existing.name,
    role || existing.role,
    company || existing.company,
    bio || existing.bio,
    JSON.stringify(skills || JSON.parse(existing.skills)),
    JSON.stringify(looking_for || JSON.parse(existing.looking_for)),
    JSON.stringify(can_help_with || JSON.parse(existing.can_help_with)),
    JSON.stringify(domains || JSON.parse(existing.domains)),
    linkedin_url !== undefined ? linkedin_url : existing.linkedin_url,
    req.params.id
  );

  const updated = serialize(db.prepare("SELECT * FROM profiles WHERE id = ?").get(req.params.id));
  syncRag(updated);
  res.json(updated);
});

module.exports = router;
