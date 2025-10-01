#!/bin/bash
set -e

echo "Stopping MaskTerial application..."
cd /opt/MaskTerial
docker compose -f docker-compose.prod.yml down --remove-orphans || true
echo "Application stopped"

