const Database = require("better-sqlite3");
const path = require("path");

const DB_PATH = path.join(__dirname, "mingle.db");

const db = new Database(DB_PATH);

// Enable WAL mode for better concurrent read performance
db.pragma("journal_mode = WAL");

db.exec(`
  CREATE TABLE IF NOT EXISTS profiles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    role TEXT NOT NULL,
    company TEXT NOT NULL,
    bio TEXT NOT NULL,
    skills TEXT NOT NULL,
    looking_for TEXT NOT NULL,
    can_help_with TEXT NOT NULL,
    domains TEXT NOT NULL,
    linkedin_url TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  );

  CREATE TABLE IF NOT EXISTS network (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_user_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    saved_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(owner_user_id, profile_id)
  );
`);

module.exports = db;
