#!/usr/bin/env bash
set -euo pipefail

# One-time VPS setup for skating-biomechanics-ml
# Run this on the VPS: bash setup-vps.sh <domain>

DOMAIN="${1:?Usage: bash setup-vps.sh <domain>}"
APP_DIR="/opt/skating-app"

echo "=== Setting up skating-biomechanics-ml on VPS ==="

# Install Docker if not present
if ! command -v docker &>/dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    usermod -aG docker "$(whoami)"
    echo "Docker installed. Log out and back in for group changes."
fi

# Install Docker Compose plugin
if ! docker compose version &>/dev/null; then
    echo "Installing Docker Compose plugin..."
    apt-get update && apt-get install -y docker-compose-plugin
fi

# Create app directory
mkdir -p "$APP_DIR"
cd "$APP_DIR"

# Create .env.prod from template
if [ ! -f .env.prod ]; then
    cat > .env.prod <<ENVEOF
# === Database ===
POSTGRES_DB=skating_ml
POSTGRES_USER=skating
POSTGRES_PASSWORD=CHANGE_ME_GENERATE_WITH_openssl_rand_hex_32

# === Backend ===
DATABASE_URL=postgresql+asyncpg://skating:CHANGE_ME@postgres:5432/skating_ml
VALKEY_URL=redis://valkey:6379/0
JWT_SECRET=CHANGE_ME_GENERATE_WITH_openssl_rand_hex_32

# === R2 Storage ===
R2_ENDPOINT_URL=
R2_ACCESS_KEY_ID=
R2_SECRET_ACCESS_KEY=
R2_BUCKET=

# === Vast.ai (optional, enables GPU dispatch) ===
VASTAI_API_KEY=

# === Caddy ===
DOMAIN=${DOMAIN}
ENVEOF
    echo "Created .env.prod — EDIT IT before deploying!"
fi

echo ""
echo "=== Setup complete ==="
echo "1. Edit $APP_DIR/.env.prod with your secrets"
echo "2. Run: cd $APP_DIR && docker compose -f compose.prod.yaml up -d"
echo "3. Caddy will auto-provision SSL for $DOMAIN"
