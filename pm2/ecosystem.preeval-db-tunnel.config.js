// Reverse tunnel, run on the central box: exposes its localhost-only ingot Postgres on the
// pre-eval box's localhost, same port. INGOT_PREEVAL_SSH and INGOT_POSTGRES_PORT from .env.
const { tunnel } = require("./tunnel");

module.exports = {
  apps: [tunnel("ingot-preeval-db-tunnel", "-R", "INGOT_PREEVAL_SSH", "INGOT_POSTGRES_PORT")],
};
