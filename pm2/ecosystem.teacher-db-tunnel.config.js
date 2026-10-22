// Reverse tunnel, run on the central box: exposes its localhost-only ingot Postgres on the teacher
// box's localhost, same port, for the teacher server's workers row. INGOT_TEACHER_SSH and
// INGOT_POSTGRES_PORT from .env.
const { tunnel } = require("./tunnel");

module.exports = {
  apps: [tunnel("ingot-teacher-db-tunnel", "-R", "INGOT_TEACHER_SSH", "INGOT_POSTGRES_PORT")],
};
