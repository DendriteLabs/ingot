"""Every block: metagraph snapshot, swap detection, private-store signals, and every few minutes the
registration price; then the access controller until it has nothing to do. Every verified upload
ends as a SUBMITTED row."""

from __future__ import annotations

import time
from typing import Any

from loguru import logger as log

from chain_reader import config, controller, prices, signals
from common import chain, db, guard
from common.config import Settings, chain_spec, r2_settings, settings
from common.loop import forever


class ChainReader:
    def __init__(
        self, conn: db.Conn, subtensor: Any, deps: controller.Deps, s: Settings, worker: str
    ) -> None:
        self.conn, self.subtensor, self.deps, self.s, self.worker = conn, subtensor, deps, s, worker
        self.last_block: int | None = None
        self.next_sweep = 0.0
        self.next_quota = 0.0

    def step(self) -> bool:
        db.heartbeat(self.conn, self.worker)
        block = chain.current_block(self.subtensor)
        if block != self.last_block:
            self.scan(block)
            self.last_block = block
            try:
                prices.sample(self.conn, self.subtensor, self.s.netuid, block)
            except Exception:
                # the price only feeds a chart; it must never hold up the signals scan
                log.exception("registration price sample failed at block {}", block)
        if time.monotonic() >= self.next_quota:
            controller.sweep_credentialed(self.conn, self.deps)
            self.next_quota = time.monotonic() + config.QUOTA_INTERVAL_S
        if time.monotonic() >= self.next_sweep:
            controller.sweep(self.conn, self.deps)
            self.next_sweep = time.monotonic() + config.SWEEP_INTERVAL_S
        return controller.tick(self.conn, self.deps)

    def scan(self, block: int) -> None:
        at = max(block - config.SCAN_LAG_BLOCKS, 0)
        snapshot, owners = chain.metagraph_snapshot(self.subtensor, self.s.netuid, at)
        swaps = chain.find_swaps(guard.uid_state(self.conn), snapshot)
        if swaps:
            confirmed = chain.confirm_swaps(self.subtensor, swaps, at)
            guard.record_swaps(self.conn, confirmed, self.s.netuid, block)
        guard.refresh_neurons(self.conn, snapshot, owners)
        uids = {hotkey: uid for uid, hotkey, _ in snapshot}
        found = chain.scan_signals(
            self.subtensor,
            self.s.netuid,
            uids=uids,
            owners=owners,
            at_block=at,
            start_block=self.s.start_block,
        )
        applied = signals.handle(self.conn, found)
        log.info("block={} signals={} applied={}", block, len(found), applied)


def main() -> None:
    s, spec = settings(), chain_spec()
    conn = db.connect()
    subtensor = chain.connect(s.network)
    deps = controller.build_deps(r2_settings(), spec)
    if s.start_block:
        commits = [
            *chain.iter_revealed(subtensor, s.netuid),
            *chain.iter_committed(subtensor, s.netuid),
        ]
        guard.record_legacy(conn, commits, s.start_block)
    worker = db.register_worker(conn, "chain_reader")
    log.info("chain reader started — netuid={} network={}", s.netuid, s.network)
    forever(
        ChainReader(conn, subtensor, deps, s, worker).step,
        idle_s=config.POLL_INTERVAL_S,
        on_stop=lambda: db.offline(conn, worker),
    )
