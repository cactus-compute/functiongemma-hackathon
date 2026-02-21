const express = require("express");
const axios = require("axios");
const multer = require("multer");
const path = require("path");
const fs = require("fs");

const router = express.Router();
const AI_SERVER = process.env.AI_SERVER_URL || "http://localhost:8001";

// Store voice notes in memory (or temp directory)
const upload = multer({ dest: "/tmp/voice-notes/" });

// POST /api/voicenotes/process - Upload voice note and trigger agentic flow
router.post("/process", upload.single("audio"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No audio file provided" });
    }

    const audioPath = req.file.path;
    
    // Read audio file as base64
    const audioBuffer = fs.readFileSync(audioPath);
    const audioBase64 = audioBuffer.toString("base64");
    
    // Send to AI server for processing
    const response = await axios.post(`${AI_SERVER}/ai/process-voice-note`, {
      audio_base64: audioBase64,
      mime_type: req.file.mimetype || "audio/webm",
    }, {
      timeout: 120000, // 2 minutes
    });
    
    // Clean up temp file
    fs.unlinkSync(audioPath);
    
    res.json(response.data);
  } catch (err) {
    console.error("Voice note processing error:", err);
    const status = err.response?.status || 500;
    const message = err.response?.data?.detail || err.message;
    res.status(status).json({ error: message });
  }
});

module.exports = router;
