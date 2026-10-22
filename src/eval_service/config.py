# The commit of the PrismML llama.cpp fork that runs q2 (gguf) models: llama-prism is
# checked out here, so a bump moves both together.
LLAMA_CPP_COMMIT = "e3de715222b2667c54a267441159a5729f11155e"  # tag prism-b10777-e3de715
# A gguf model reads the teacher's answer in llama-states (scripts/build_llama.sh builds both),
# LLAMA_PER_GPU processes per GPU, each one request at a time, in passes of LLAMA_BATCH tokens.
LLAMA_STATES = "build/llama_states/llama-states"
LLAMA_BATCH = 2048
# A read spends most of its time off the GPU, writing its ~0.7 GB states file: one process kept a
# GPU 10-20% busy, so a second one about doubles the reads a GPU gets through. Each holds ~12 GB.
LLAMA_PER_GPU = 2
# A read that takes longer than LLAMA_READ_S plus LLAMA_S_PER_1K per thousand tokens means a hung
# process (genesis reads ~3k tokens/s); it is stopped and blamed on its GPU's model.
LLAMA_READ_S = 120
LLAMA_S_PER_1K = 10

LEASE_S = 4 * 3600
MAX_ATTEMPTS = 3
IDLE_S = 5.0

SAMPLES_PER_DOMAIN = 60
# Threads a draw renders its long candidates with, and the corpus is read with at startup.
DRAW_THREADS = 32
# A win needs the challenger closer to the teacher than the king on both measures, each by this
# share of the king's own score in that eval, so the bar shrinks as kings near the teacher. The two
# move together (distance by ~0.56% per 1% of KL across the duels so far), hence their ratio.
KL_MARGIN = 0.015  # of the king's mean KL(teacher ‖ model) per answer token
DIST_MARGIN = 0.0084  # of the king's mean |h - h_teacher| / |h_teacher| over LAYERS

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
# Samples of the next eval the teacher answers while the current one drains (run.Prefetch): once
# every sample is admitted, the teacher's servers empty out over the last long answers, about
# the final fifth of an eval. The answers' state files wait in /dev/shm, ~0.7 GB per sample.
PREFETCH_SAMPLES = 100
# A sample that fails is left out and the eval goes on. More than this many is a broken eval, or
# a model failing on purpose to shed hard prompts, and ends the eval as every failure once did.
MAX_FAILED_SAMPLES = 10
PROGRESS_CHUNKS = 20  # how often a running eval writes samples_done for the dashboard
# A state file no eval fetched in this long was abandoned (its eval died); the teacher server
# sweeps such files, /dev/shm being RAM.
STATES_TTL_S = 3600
# how often the teacher server touches its workers row
TEACHER_HEARTBEAT_S = 30
# how often a running eval logs its per-stage timings (eval_service.timing)
TIMING_REPORT_S = 60.0
