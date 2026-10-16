# AGENTS.md

How agents develop Ingot. `llms.txt` is the miner-facing description of the network; keep
the two apart. The code and `schema.sql` are the map; this file does not repeat them.

## How to work here

- KISS. The simplest thing that works wins; clever code is a cost, not a feature.
- YAGNI. Build the step the pipeline needs now; no options, layers or abstractions for later.
- DRY. Every piece of knowledge has one home. Before writing a function, look for the one
  that already does it: `common/`, `private_store/`, the other services. Call it, extend
  it, or move it to `common/` if a second service needs it. Never write a second version of
  something that exists.
- Prefer the one-liner, the standard library, and the dependency we already have.
- One pm2 app per service folder, each a loop around `run_once()` in its `main.py`, with its
  tuning constants in its `config.py`. Services share code through `common/`, never through
  each other.
- All coordination is `submissions.state`: `db.claim` a row, do one thing, `db.finish` it.
  A crash goes through `db.fail`; leases make a dead worker's row claimable again.
- Raw SQL with psycopg; no ORM, no migration tool. `schema.sql` is idempotent.
- `.env` holds secrets and deployment values only. A number that tunes a rule goes in the
  service's `config.py`; something miners must agree on goes in `chain.toml`.
- Ruff and ty are the style guide. Don't argue with them, don't add noqa to win.
- Readability first. Code is read far more often than it is written. Optimize for the next person reading it.
- Follow OOP principles and keep the code modular and easily extensible.

## Comments

A comment is for the next agent that opens the file: compact, and only where it saves them
from re-deriving or re-breaking something. A decision and its reason, a trap, an open
question, a reference. If the next agent would not act differently for having read it, leave
it out. Most code has none. Decisions that are non-trivial or seem illogical should be 
documented with a reason.

- A module docstring is one or two sentences on what the module is for. A function or class
  gets a docstring only when its name and signature don't already say it; one line, no
  `Args`/`Returns` sections.
- An inline comment is one line above the code, in plain words, saying *why*. Never what
  the next line does, never restating a name, never a date, an author or a changelog.
- No commented-out code; git remembers it.
- As short as it needs to be and no shorter: cut every word that does not change what the
  reader does next. One line is the norm; a paragraph needs a reason.

## Tests

- A test exists to catch a broken pipeline, not to prove a function runs. `test_pipeline.py`
  drives `run_once` of each service through the docker-compose Postgres with fakes at the
  boundary (chain, R2, Hugging Face, the GPU).
- Pure rules (incentive, validation, the win rule) get a direct, table-driven test.
- Tests of helpers, of mocks, or that a function returns a dict are not worth their run
  time. Don't add them; delete them when you find them.
- A test that needs a GPU or the network carries the `gpu` or `network` marker.

## Docs

- `README.md`: how to develop and mine. `llms.txt`: how the network works, for miners.
  This file: how we work. No other docs files.
- Changing a rule or a setting miners can see updates `llms.txt` in the same change.
