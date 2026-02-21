const express = require("express");
const cors = require("cors");

const profilesRouter = require("./routes/profiles");
const networkRouter = require("./routes/network");
const qrRouter = require("./routes/qr");
const outreachRouter = require("./routes/outreach");

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

app.use("/api/profiles", profilesRouter);
app.use("/api/network", networkRouter);
app.use("/api/qr", qrRouter);
app.use("/api/outreach", outreachRouter);

app.get("/health", (_req, res) => res.json({ status: "ok" }));

app.listen(PORT, () => {
  console.log(`Mingle backend running on http://localhost:${PORT}`);
});
