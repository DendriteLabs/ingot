// Forward tunnels, run on the eval box: bring the teacher box's ingot-teacher-server to the eval
// box's localhost (its INGOT_TEACHER_URL). INGOT_TEACHER_SSH, INGOT_TEACHER_PORT and
// INGOT_TEACHER_TUNNELS from .env. The teacher's hidden states come through them: N tunnels listen
// on INGOT_TEACHER_PORT..+N-1, all reaching the server's INGOT_TEACHER_PORT.
const { tunnel, loadEnv } = require("./tunnel");

const count = Number(loadEnv().INGOT_TEACHER_TUNNELS || 1);

module.exports = {
  apps: Array.from({ length: count }, (_, i) =>
    tunnel(i ? `ingot-teacher-tunnel-${i}` : "ingot-teacher-tunnel", "-L", "INGOT_TEACHER_SSH",
      "INGOT_TEACHER_PORT", i),
  ),
};
