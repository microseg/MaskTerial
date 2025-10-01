#!/bin/bash
set -e

echo "Preparing for installation..."

# Install Docker Compose if not present
if ! docker compose version >/dev/null 2>&1; then
  echo "Installing Docker Compose..."
  mkdir -p /usr/local/lib/docker/cli-plugins
  curl -SL https://github.com/docker/compose/releases/download/v2.29.2/docker-compose-linux-x86_64 -o /usr/local/lib/docker/cli-plugins/docker-compose
  chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
fi

docker compose version

# Backup data directory (preserve user data)
if [ -d /opt/MaskTerial/data ]; then
  echo "Backing up data directory..."
  cp -r /opt/MaskTerial/data /tmp/maskterial-data-backup || true
fi

# Clean up old installation (CodeDeploy will copy new files here)
echo "Cleaning up old installation..."
rm -rf /opt/MaskTerial/* /opt/MaskTerial/.* 2>/dev/null || true

# Restore data directory
if [ -d /tmp/maskterial-data-backup ]; then
  echo "Restoring data directory..."
  mkdir -p /opt/MaskTerial/data
  cp -r /tmp/maskterial-data-backup/* /opt/MaskTerial/data/ || true
  rm -rf /tmp/maskterial-data-backup
fi

# ECR Login
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(ec2-metadata --availability-zone | sed 's/[a-z]$//' | awk '{print $2}')
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

echo "Preparation completed"

