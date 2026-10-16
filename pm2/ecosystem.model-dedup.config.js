// Pre-eval box: fingerprints valid submissions; OpenSearch from scripts/setup_opensearch.sh.
const path = require("path");

module.exports = {
  apps: [
    {
      name: "ingot-model-dedup",
      cwd: path.resolve(__dirname, ".."),
      script: ".venv/bin/ingot-model-dedup",
      interpreter: "none",
      env: { PYTHONUNBUFFERED: "1" },
      autorestart: true,
      max_restarts: 20,
      restart_delay: 5000,
      time: true,
    },
  ],
};
