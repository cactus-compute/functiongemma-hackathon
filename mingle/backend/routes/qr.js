const express = require("express");
const QRCode = require("qrcode");

const router = express.Router();

const FRONTEND_BASE = process.env.FRONTEND_URL || "http://localhost:5173";

// GET /api/qr/:profileId
router.get("/:profileId", async (req, res) => {
  const url = `${FRONTEND_BASE}/profile/${req.params.profileId}`;
  try {
    const dataUrl = await QRCode.toDataURL(url, { width: 300, margin: 2 });
    res.json({ qr: dataUrl, url });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
