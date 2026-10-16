"""vLLM logits processor that forces a request along a given token sequence; the eval's servers
load it so the king and the challenger read exactly the teacher's answer."""

from __future__ import annotations

import torch
from vllm.v1.worker.gpu.sample.logits_processor import LogitsProcessor

KEY = "force_token_ids"


class TrajectoryForcer(LogitsProcessor):
    """A request with `vllm_xargs.force_token_ids` (prompt followed by answer) can only produce
    the token that follows its current position. It runs after the forward pass, so hidden states
    and raw distributions stay the model's own; only the next input token is fixed."""

    def __init__(self, vllm_config, req_states):
        # [slot, absolute position] -> forced token id, -1 where free
        self.traj = torch.full(
            (req_states.max_num_reqs, vllm_config.model_config.max_model_len + 1),
            -1,
            dtype=torch.int64,
            device=req_states.device,
        )
        self.staged: dict[int, list[int] | None] = {}

    @classmethod
    def validate_params(cls, sampling_params):
        ids = (sampling_params.extra_args or {}).get(KEY)
        if ids is not None and not (isinstance(ids, list) and all(isinstance(t, int) for t in ids)):
            raise ValueError(f"{KEY} must be a list of int token ids")

    def add_request(self, req_idx, sampling_params):
        ids = (sampling_params.extra_args or {}).get(KEY)
        self.staged[req_idx] = ids
        return ids is not None

    def apply_staged_writes(self):
        for slot, ids in self.staged.items():
            self.traj[slot].fill_(-1)
            if ids:
                self.traj[slot, : len(ids)] = torch.tensor(ids, dtype=torch.int64)
        self.staged.clear()

    def apply(self, logits, ctx):
        target = self.traj[ctx.expanded_idx_mapping.long(), ctx.pos.long() + 1]
        rows = torch.nonzero(target >= 0).squeeze(1)
        if rows.numel() == 0:
            return logits
        cols = target[rows]
        kept = logits[rows, cols].clone()
        logits[rows] = float("-inf")
        logits[rows, cols] = kept
        return logits
