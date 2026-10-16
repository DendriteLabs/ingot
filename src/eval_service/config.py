LEASE_S = 4 * 3600
MAX_ATTEMPTS = 3
IDLE_S = 5.0

SAMPLES_PER_DOMAIN = 60
# The challenger must beat the king on both, each by its own margin: about 3% of genesis's own
# KL (0.0033) and distance (0.040). King and challenger read the same answers, so at ~1000
# samples the paired difference has an SE of ~0.00002 KL and ~0.00006 distance: an equal model
# clears a margin about once in 10,000 evals, a model 5.7% / 7.1% closer (fp8_dynamic) always.
KL_MARGIN = 0.0001  # mean KL(teacher ‖ model) per answer token, nats
DIST_MARGIN = 0.0012  # mean |h - h_teacher| / |h_teacher| over LAYERS

# One vLLM server per GPU. The comparison math runs on the challenger's GPUs, whose servers
# leave room for it (an lm_head copy and a few GiB of work space).
MEMORY_SHARE = {"teacher": 0.88, "king": 0.88, "challenger": 0.80}
BASE_PORT = 9100
SERVER_START_S = 3600
# Hidden states are compared at every 8th layer; the last one also gives the next-token
# log-probs. Each layer costs cache: 64 layers left room for 3 requests per GPU, 8 for ~30.
LAYERS = tuple(range(8, 65, 8))

MAX_TOKENS = 4096
# prompt and answer together; the draw passes over prompts that leave less than MAX_TOKENS
MAX_MODEL_LEN = 32768
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
# Every Nth sample the teacher reads its own answer again: the noise floor batching adds.
RERUN_EVERY = 10
# Samples in flight. Each server's share is bounded by its cache (_hold) and --max-num-seqs; at
# 128 every GPU ran about 13 requests with an empty queue, the teacher being the slowest stage.
WORKERS = 384
# A sample that fails is left out and the eval goes on. More than this many is a broken eval, or
# a model failing on purpose to shed hard prompts, and ends the eval as every failure once did.
MAX_FAILED_SAMPLES = 10
# A state file no eval fetched in this long was abandoned (its eval died); the teacher server
# sweeps such files, /dev/shm being RAM.
STATES_TTL_S = 3600
# how often a running eval logs its per-stage timings (eval_service.timing)
TIMING_REPORT_S = 60.0

KEEP_FAILED = 3
KEEP_KINGS = 2
