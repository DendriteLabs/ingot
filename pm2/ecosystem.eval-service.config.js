// Eval box (8 GPUs): runs the duels against the central Postgres (ingot-eval-db-tunnel).
const path = require("path");

module.exports = {
  apps: [
    {
      name: "ingot-eval-service",
      cwd: path.resolve(__dirname, ".."),
      script: ".venv/bin/ingot-eval-service",
      interpreter: "none",
      env: { PYTHONUNBUFFERED: "1" },
      autorestart: true,
      max_restarts: 20,
      restart_delay: 5000,
      time: true,
    },
  ],
};
