// Reverse tunnel, run on the central box: exposes its localhost-only ingot Postgres on the eval
// box's localhost, same port. INGOT_EVAL_SSH and INGOT_POSTGRES_PORT from .env.
const { tunnel } = require("./tunnel");

module.exports = {
  apps: [tunnel("ingot-eval-db-tunnel", "-R", "INGOT_EVAL_SSH", "INGOT_POSTGRES_PORT")],
};
