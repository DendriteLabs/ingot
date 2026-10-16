LEASE_S = 3600
MAX_ATTEMPTS = 3
IDLE_S = 5.0

NEAREST_K = 10  # banked fingerprints compared in full, picked by their summary vectors
# Thresholds on the safe side of the Qwen3.8 calibration (qwantbedo docs/DEDUP_CALIBRATION.md).
# Correlation of two quantization errors at or above which one model copies the other. Honest
# recipes <= 0.50 (0.60 for GPTQ on the same data); the only copy better than plain RTN: 0.82.
# Set above the calibrated 0.7 / 0.8 for now, so that a re-quantization of genesis (corr 0.993
# with genesis, per-tensor up to 0.999) passes: only near-exact copies are caught until tightened.
COPY_CORR = 0.995
# A tensor counts as copied at this correlation of its own: copied tensors sit near 1.0, honest
# GPTQ pairs never reach it. Plain RTN does: it is the same tensor whoever runs it.
PART_CORR = 0.999
# Share of the error energy in copied tensors that makes a partial copy: ~20 % of the layers.
COPIED_FRAC = 0.2

KEEP_DUPLICATES = 3
# A UNIQUE download stays this long for a look, then goes; the eval box has its own copy.
KEEP_UNIQUE_S = 15 * 60
