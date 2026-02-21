const express = require("express");
const axios = require("axios");

const router = express.Router();

const AI_SERVER = process.env.AI_SERVER_URL || "http://localhost:8001";

// POST /api/outreach/rank  — proxies to FastAPI /ai/rank-contacts
router.post("/rank", async (req, res) => {
  try {
    const response = await axios.post(`${AI_SERVER}/ai/rank-contacts`, req.body, {
      timeout: 60000,
    });
    res.json(response.data);
  } catch (err) {
    const status = err.response?.status || 502;
    const message = err.response?.data?.detail || err.message;
    res.status(status).json({ error: message });
  }
});

// POST /api/outreach/draft  — proxies to FastAPI /ai/draft-outreach
router.post("/draft", async (req, res) => {
  try {
    const response = await axios.post(`${AI_SERVER}/ai/draft-outreach`, req.body, {
      timeout: 60000,
    });
    res.json(response.data);
  } catch (err) {
    const status = err.response?.status || 502;
    const message = err.response?.data?.detail || err.message;
    res.status(status).json({ error: message });
  }
});

module.exports = router;
