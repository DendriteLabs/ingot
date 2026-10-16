"""Submission states. A service claims rows in one state and leaves them in the next.

    SUBMITTED → VALIDATING → VALID → DEDUPING → UNIQUE → EVALUATING → WON → CROWNING → CROWNED
                           ↘ INVALID           ↘ DUPLICATE          ↘ LOST
Any running stage that raises max_attempts times ends in FAILED.
"""

SUBMITTED = "SUBMITTED"  # verified private upload, waiting for the validator
VALIDATING = "VALIDATING"
VALID = "VALID"  # waiting for the dedup
INVALID = "INVALID"  # terminal; `error` says why
DEDUPING = "DEDUPING"
UNIQUE = "UNIQUE"  # its weights copy no banked model; waiting for the evaluator
DUPLICATE = "DUPLICATE"  # terminal; `error` names the model it copies
EVALUATING = "EVALUATING"
WON = "WON"  # beat the king, waiting for the promoter
LOST = "LOST"  # terminal
CROWNING = "CROWNING"
CROWNED = "CROWNED"  # terminal; in the kings table and mirrored to Hugging Face
FAILED = "FAILED"  # terminal; a stage kept crashing, `error` says where

TERMINAL = frozenset({INVALID, DUPLICATE, LOST, CROWNED, FAILED})
