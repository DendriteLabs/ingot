// Forward tunnel, run on the eval box: brings the teacher box's ingot-teacher-server to the eval
// box's localhost, same port (its INGOT_TEACHER_URL). INGOT_TEACHER_SSH and INGOT_TEACHER_PORT from
// .env. The teacher's hidden states come through it: hundreds of MB per sample.
const { tunnel } = require("./tunnel");

module.exports = {
  apps: [tunnel("ingot-teacher-tunnel", "-L", "INGOT_TEACHER_SSH", "INGOT_TEACHER_PORT")],
};
