LEASE_S = 3600
MAX_ATTEMPTS = 3
IDLE_S = 5.0

NEAREST_K = 10  # banked fingerprints compared in full, picked by their summary vectors
COPY_CORR = 0.9999  # correlation of the two errors over the whole model, at or above
PART_CORR = 0.9999
COPIED_FRAC = 0.99

KEEP_DUPLICATES = 3
# A UNIQUE download stays this long for a look, then goes; the eval box has its own copy.
KEEP_UNIQUE_S = 15 * 60
