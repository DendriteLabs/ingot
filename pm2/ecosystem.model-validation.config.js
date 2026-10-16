// Pre-eval box: validates submissions against the central Postgres (ingot-preeval-db-tunnel).
const path = require("path");

module.exports = {
  apps: [
    {
      name: "ingot-model-validation",
      cwd: path.resolve(__dirname, ".."),
      script: ".venv/bin/ingot-model-validation",
      interpreter: "none",
      env: { PYTHONUNBUFFERED: "1" },
      autorestart: true,
      max_restarts: 20,
      restart_delay: 5000,
      time: true,
    },
  ],
};
