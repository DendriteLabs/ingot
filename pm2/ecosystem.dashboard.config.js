// Central box: keeps the site bucket's api/ (state.json, each eval's page) current; the site
// itself is static, pushed with scripts/push_site.py.
const path = require("path");

module.exports = {
  apps: [
    {
      name: "ingot-dashboard",
      cwd: path.resolve(__dirname, ".."),
      script: ".venv/bin/ingot-dashboard",
      interpreter: "none",
      env: { PYTHONUNBUFFERED: "1" },
      autorestart: true,
      max_restarts: 20,
      restart_delay: 5000,
      time: true,
    },
  ],
};
