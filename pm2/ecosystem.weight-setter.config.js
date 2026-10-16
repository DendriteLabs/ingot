// Central box: sets the validator's weights for the reign.
const path = require("path");

module.exports = {
  apps: [
    {
      name: "ingot-weight-setter",
      cwd: path.resolve(__dirname, ".."),
      script: ".venv/bin/ingot-weight-setter",
      interpreter: "none",
      env: { PYTHONUNBUFFERED: "1" },
      autorestart: true,
      max_restarts: 20,
      restart_delay: 5000,
      time: true,
    },
  ],
};
