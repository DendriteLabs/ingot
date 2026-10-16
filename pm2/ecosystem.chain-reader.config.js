// Central box: reads the chain, backfills used hotkeys, issues upload credentials.
const path = require("path");

module.exports = {
  apps: [
    {
      name: "ingot-chain-reader",
      cwd: path.resolve(__dirname, ".."),
      script: ".venv/bin/ingot-chain-reader",
      interpreter: "none",
      env: { PYTHONUNBUFFERED: "1" },
      autorestart: true,
      max_restarts: 20,
      restart_delay: 5000,
      time: true,
    },
  ],
};
