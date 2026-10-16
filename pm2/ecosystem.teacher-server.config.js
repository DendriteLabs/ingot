// Pre-eval box: the teacher for an eval box without teacher GPUs, on INGOT_TEACHER_GPUS, served on
// 127.0.0.1:INGOT_TEACHER_PORT for ingot-teacher-tunnel.
const path = require("path");

module.exports = {
  apps: [
    {
      name: "ingot-teacher-server",
      cwd: path.resolve(__dirname, ".."),
      script: ".venv/bin/ingot-teacher-server",
      interpreter: "none",
      env: { PYTHONUNBUFFERED: "1" },
      autorestart: true,
      max_restarts: 20,
      restart_delay: 5000,
      time: true,
    },
  ],
};
