#!/bin/bash
set -e

echo "Validating MaskTerial service..."

# Wait for service to be healthy (up to 2 minutes)
for i in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8000/ >/dev/null 2>&1; then
    echo "✓ Service is healthy"
    docker ps
    docker images --digests | grep material-recognition-service | head -1
    exit 0
  fi
  echo "Waiting for service... ($i/60)"
  sleep 2
done

echo "✗ Service validation failed"
docker compose -f /opt/MaskTerial/docker-compose.prod.yml logs
exit 1

