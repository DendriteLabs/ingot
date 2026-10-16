#!/usr/bin/env bash
# The dedup bank's OpenSearch: start it in its own container and wait until it is healthy.
# --seed then fingerprints genesis into it (GPU box, ~10 min), generating INGOT_DEDUP_SECRET
# in .env first if it is empty.
#
#   scripts/setup_opensearch.sh [--seed]
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f .env ] || { echo "no .env: copy .env.example and fill it in" >&2; exit 1; }

PORT="$(sed -n 's/^INGOT_OPENSEARCH_PORT=//p' .env)"
PORT="${PORT:-9270}"
NAME="${INGOT_OS_CONTAINER:-ingot_opensearch}"
IMAGE="${INGOT_OS_IMAGE:-opensearchproject/opensearch:2.19.1}"
VOLUME="${INGOT_OS_VOLUME:-ingot_opensearch_data}"
HEAP="${INGOT_OS_HEAP:-8g}"

# no auth or TLS, so it only ever listens on localhost; the bank survives on the named volume
docker volume create "$VOLUME" >/dev/null
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --restart unless-stopped \
    -p "127.0.0.1:$PORT:9200" \
    -v "$VOLUME:/usr/share/opensearch/data" \
    -e discovery.type=single-node \
    -e DISABLE_SECURITY_PLUGIN=true \
    -e DISABLE_INSTALL_DEMO_CONFIG=true \
    -e "OPENSEARCH_JAVA_OPTS=-Xms$HEAP -Xmx$HEAP" \
    "$IMAGE" >/dev/null

for _ in $(seq 1 300); do
    curl -fs "http://127.0.0.1:$PORT/_cluster/health" >/dev/null 2>&1 && break
    sleep 2
done
curl -fs "http://127.0.0.1:$PORT/_cluster/health" >/dev/null \
    || { docker logs --tail 30 "$NAME" >&2; exit 1; }

if [ "${1:-}" != "--seed" ]; then
    echo "OpenSearch is up; seed genesis on the GPU box with: scripts/setup_opensearch.sh --seed"
    exit 0
fi

# the bank is fingerprinted under this secret, so one that is set is never replaced
if ! grep -q '^INGOT_DEDUP_SECRET=.' .env; then
    grep -q '^INGOT_DEDUP_SECRET=' .env || echo 'INGOT_DEDUP_SECRET=' >>.env
    sed -i "s/^INGOT_DEDUP_SECRET=.*/INGOT_DEDUP_SECRET=$(openssl rand -hex 32)/" .env
    echo "generated INGOT_DEDUP_SECRET in .env; model-dedup must run with the same one"
fi
uv run --extra dedup scripts/seed_dedup_bank.py
