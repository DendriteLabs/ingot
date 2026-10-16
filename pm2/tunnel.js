// Builds an SSH tunnel app from .env, so no host or port is committed: the target is an ssh alias
// or user@host from .env, the forwarded port one of the .env ports.
const fs = require("fs");
const path = require("path");

function loadEnv() {
  const envPath = path.resolve(__dirname, "..", ".env");
  const env = {};
  if (!fs.existsSync(envPath)) return env;
  for (const line of fs.readFileSync(envPath, "utf8").split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const index = trimmed.indexOf("=");
    if (index === -1) continue;
    env[trimmed.slice(0, index)] = trimmed.slice(index + 1);
  }
  return env;
}

// direction "-R" (run on the central box, pushes its port to the target) or "-L" (pulls the
// target's port here); hostKey and portKey name the .env entries
function tunnel(name, direction, hostKey, portKey) {
  const env = loadEnv();
  const host = env[hostKey];
  const port = env[portKey];
  if (!host || !port) throw new Error(`${name}: set ${hostKey} and ${portKey} in .env`);
  return {
    name,
    script: "ssh",
    args: [
      "-N",
      "-o", "ExitOnForwardFailure=yes",
      "-o", "ServerAliveInterval=30",
      "-o", "ServerAliveCountMax=3",
      "-o", "BatchMode=yes",
      // the teacher's hidden states are hundreds of MB per sample; AES-GCM moved 981 MB/s on one
      // stream between the boxes, the default chacha20 852
      "-c", "aes128-gcm@openssh.com",
      direction, `127.0.0.1:${port}:127.0.0.1:${port}`,
      host,
    ],
    autorestart: true,
    restart_delay: 5000,
  };
}

module.exports = { tunnel };
