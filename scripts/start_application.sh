#!/bin/bash
set -e

echo "Starting MaskTerial application..."
cd /opt/MaskTerial

# Core 3-line deployment
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up -d

echo "Application started"

