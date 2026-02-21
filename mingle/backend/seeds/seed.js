const fs = require("fs");
const path = require("path");
const db = require("../db");
const { v4: uuidv4 } = require("uuid");

const usersPath = path.join(__dirname, "users.json");
const users = JSON.parse(fs.readFileSync(usersPath, "utf-8"));

// Clear existing profiles
db.prepare("DELETE FROM profiles").run();

// Insert seed users
const stmt = db.prepare(`
  INSERT INTO profiles (id, name, role, company, bio, skills, looking_for, can_help_with, domains, linkedin_url)
  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
`);

for (const user of users) {
  const id = uuidv4();
  stmt.run(
    id,
    user.name,
    user.role,
    user.company,
    user.bio,
    JSON.stringify(user.skills || []),
    JSON.stringify(user.looking_for || []),
    JSON.stringify(user.can_help_with || []),
    JSON.stringify(user.domains || []),
    user.linkedin_url || ""
  );
  console.log(`Created: ${user.name} (${user.role})`);
}

console.log(`\nSeeded ${users.length} users.`);
