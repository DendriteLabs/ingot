# The commit of the Dendrite llama.cpp fork (DendriteLabs/llama.cpp, PrismML's fork plus our
# changes) that runs q2 (gguf) models: llama-prism is checked out here, so a bump moves both
# together.
LLAMA_CPP_COMMIT = "62c418ccd7aefee37997e94c2d9fac1eed7fc89b"
# A gguf model reads the teacher's answer in llama-states (scripts/build_llama.sh builds both),
# LLAMA_PER_GPU processes per GPU, each one request at a time, in passes of LLAMA_BATCH tokens.
LLAMA_STATES = "build/llama_states/llama-states"
LLAMA_BATCH = 2048
# llama-states processes per GPU per role: a read mostly writes its states file, so several
# share a GPU
LLAMA_PER_GPU = 6
# A read that takes longer than LLAMA_READ_S plus LLAMA_S_PER_1K per thousand tokens is a hung
# process: it is stopped and blamed on its GPU's model.
LLAMA_READ_S = 120
LLAMA_S_PER_1K = 10

LEASE_S = 4 * 3600
MAX_ATTEMPTS = 3
IDLE_S = 5.0

SAMPLES_PER_DOMAIN = 40
DOMAIN_SAMPLES = {"terminal": 300, "swe": 250, "reasoning": 100, "science": 100, "math": 50}
# trajectories of these sources are cut later in the run (dataset._cut)
LATE_CUT_SOURCES = frozenset({"swe_traces", "term_rollouts"})
# Threads a draw renders its long candidates with, and the corpus is read with at startup.
DRAW_THREADS = 32
# A win needs the challenger closer to the teacher on KL by this share of the king's own KL in
# that eval.
KL_MARGIN = 0.03

# One vLLM server per GPU. The comparison math runs on the king's and the challenger's GPUs, whose
# servers leave room for it (an lm_head copy and a few GiB of work space). The teacher's share
# leaves room for vLLM's start-up peak above it and the eval service's own CUDA context on GPU 0.
MEMORY_SHARE = {"teacher": 0.90, "king": 0.65, "challenger": 0.65}
BASE_PORT = 9100
SERVER_START_S = 3600
# The hidden states the servers write per token: the last block's, which gives the next-token
# log-probs.
LAYERS = (64,)

MAX_TOKENS = 4096
DOMAIN_MAX_TOKENS = {"science": 12288, "math": 8192, "reasoning": 12288}
# prompt and answer together; the draw passes over prompts that leave less than MAX_TOKENS
MAX_MODEL_LEN = 36864
MAX_BYTES_PER_TOKEN = 8
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
# Every Nth sample the teacher reads its own answer again: the noise floor batching adds.
RERUN_EVERY = 10
# Samples in flight. Each server's share is bounded by its cache (_hold) and --max-num-seqs.
WORKERS = 800
# Samples of the next eval the teacher answers while the current one drains (run.Prefetch): once
# every sample is admitted, the teacher's servers empty out over the last long answers. The
# answers' state files wait in /dev/shm.
PREFETCH_SAMPLES = 100
# A sample that fails is left out and the eval goes on. More than this many is a broken eval, or
# a model failing on purpose to shed hard prompts, and ends the eval.
MAX_FAILED_SAMPLES = 10
PROGRESS_CHUNKS = 20  # how often a running eval writes samples_done for the dashboard
# A state file no eval fetched in this long was abandoned (its eval died); the teacher server
# and the eval service sweep such files, /dev/shm being RAM.
STATES_TTL_S = 3600
# of /dev/shm that written state files and the promises of requests in flight may fill
SHM_SHARE = 0.9
# how often the teacher server touches its workers row
TEACHER_HEARTBEAT_S = 30
# how often a running eval logs its per-stage timings (eval_service.timing)
TIMING_REPORT_S = 60.0
