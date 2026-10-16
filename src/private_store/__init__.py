"""Private R2 model submissions: credentials, commitments, upload verification.

Miners upload into a prefix-jailed private bucket with validator-issued temporary
credentials; upload access is revoked before any verification runs. `controller` and
`signals` run inside ingot-chain-reader; the other modules are shared with the miner CLI.
"""
