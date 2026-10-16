// Central box: crowns a won challenger and publishes it to Hugging Face.
const path = require("path");

module.exports = {
  apps: [
    {
      name: "ingot-promoter",
      cwd: path.resolve(__dirname, ".."),
      script: ".venv/bin/ingot-promoter",
      interpreter: "none",
      env: { PYTHONUNBUFFERED: "1" },
      autorestart: true,
      max_restarts: 20,
      restart_delay: 5000,
      time: true,
    },
  ],
};
