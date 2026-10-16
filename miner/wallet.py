from __future__ import annotations

import os
from typing import Any

from loguru import logger

from common import chain


def build_wallet(coldkey: str, hotkey: str) -> Any:
    return chain.build_wallet(coldkey, hotkey, os.environ.get("INGOT_WALLET_PATH", ""))


def _neuron_uid(subtensor: Any, ss58: str, netuid: int) -> int | None:
    neuron = subtensor.get_neuron_for_pubkey_and_subnet(ss58, netuid)
    if neuron is None or neuron.is_null:
        return None
    return neuron.uid


def register(coldkey: str, hotkey: str, netuid: int, network: str, *, yes: bool) -> int | None:
    wallet = build_wallet(coldkey, hotkey)
    ss58 = wallet.hotkey.ss58_address
    subtensor = chain.connect(network)
    uid = _neuron_uid(subtensor, ss58, netuid)
    if uid is not None:
        logger.info("hotkey {} already registered on netuid {} (uid {})", ss58, netuid, uid)
        return uid
    cost = subtensor.recycle(netuid)
    balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)
    logger.info("registration cost on netuid {}: {} (coldkey balance {})", netuid, cost, balance)
    if balance is not None and cost is not None and balance < cost:
        raise SystemExit(f"insufficient balance: have {balance}, need {cost}")
    print(f"about to register hotkey {hotkey} ({ss58}) on netuid {netuid} ({network}) for {cost}")
    if not yes and input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
        logger.info("aborted — not registered")
        return None
    response = subtensor.burned_register(wallet=wallet, netuid=netuid)
    if not getattr(response, "success", False):
        raise SystemExit(f"registration failed: {getattr(response, 'message', response)}")
    uid = _neuron_uid(subtensor, ss58, netuid)
    logger.info("registered hotkey {} on netuid {} — uid {}", ss58, netuid, uid)
    return uid
