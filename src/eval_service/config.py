# The commit of the Dendrite llama.cpp fork (DendriteLabs/llama.cpp, PrismML's fork plus our
# changes) that runs q2 (gguf) models: llama-prism is checked out here, so a bump moves both
# together.
LLAMA_CPP_COMMIT = "c96baf4abe24d11a50f326e40b6223b687449d63"
# A gguf model reads the teacher's answer in llama-states (scripts/build_llama.sh builds both),
# LLAMA_PER_GPU processes per GPU, each one request at a time, in passes of LLAMA_BATCH tokens.
LLAMA_STATES = "build/llama_states/llama-states"
LLAMA_BATCH = 2048
# A read spends most of its time off the GPU, writing its ~0.7 GB states file: one process kept a
# GPU 10-20% busy, and two got through an eval's reads in a quarter of its time. Four on one GPU
# per role (~12 GB each) leave the other GPUs to the teacher, the stage every eval waits on.
LLAMA_PER_GPU = 4
# A read that takes longer than LLAMA_READ_S plus LLAMA_S_PER_1K per thousand tokens means a hung
# process (genesis reads ~3k tokens/s); it is stopped and blamed on its GPU's model.
LLAMA_READ_S = 120
LLAMA_S_PER_1K = 10

LEASE_S = 4 * 3600
MAX_ATTEMPTS = 3
IDLE_S = 5.0

SAMPLES_PER_DOMAIN = 40
DOMAIN_SAMPLES = {"terminal": 300, "swe": 250, "tau3": 100, "science": 100, "math": 50}
LATE_CUT_SOURCES = frozenset({"open_swe_minisweagent", "tbpro_qwen38"})
# Threads a draw renders its long candidates with, and the corpus is read with at startup.
DRAW_THREADS = 32
# A win needs the challenger closer to the teacher on KL by this share of the king's own KL in that
# eval, so the bar shrinks as kings near the teacher. The hidden-state distance only guards: the
# challenger may not be further from the teacher than the king.
KL_MARGIN = 0.015  # of the king's mean KL(teacher ‖ model) per answer token
DIST_SLACK = 0.0  # of the king's mean |h - h_teacher| / |h_teacher| over LAYERS

# One vLLM server per GPU. The comparison math runs on the king's and the challenger's GPUs, whose
# servers leave room for it (an lm_head copy and a few GiB of work space).
MEMORY_SHARE = {"teacher": 0.88, "king": 0.80, "challenger": 0.80}
BASE_PORT = 9100
SERVER_START_S = 3600
# Hidden states are compared at every 8th layer; the last one also gives the next-token
# log-probs. Each layer costs cache: 64 layers left room for 3 requests per GPU, 8 for ~30.
LAYERS = tuple(range(8, 65, 8))

MAX_TOKENS = 4096
DOMAIN_MAX_TOKENS = {"science": 12288, "math": 8192}
# prompt and answer together; the draw passes over prompts that leave less than MAX_TOKENS. At
# 65536 a teacher GPU's KV cache held 87k tokens, one or two agent prompts: 5 requests in flight
# and a 70-minute duel (eval 158). 32k of prompt keeps most of an agent run's depth.
MAX_MODEL_LEN = 36864
MAX_BYTES_PER_TOKEN = 8
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
