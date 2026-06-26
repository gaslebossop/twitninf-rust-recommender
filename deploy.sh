#!/usr/bin/env bash
# deploy.sh — Compile en local (si Rust installé) OU compile sur le VPS
# Usage : bash deploy.sh

set -e

VPS="debian@51.210.11.74"
KEY="C:/Users/nouno/OneDrive/Bureau/Documents/privatessh"
REMOTE_DIR="/home/debian/rust-recommender"

echo "=== Envoi des sources vers le VPS ==="
tar -czf - --exclude=target --exclude=.git . \
  | ssh -i "$KEY" "$VPS" "mkdir -p $REMOTE_DIR && cd $REMOTE_DIR && tar -xzf -"

echo "=== Compilation release sur le VPS ==="
ssh -i "$KEY" "$VPS" "
  set -e
  # Installer Rust si absent
  if ! command -v cargo &>/dev/null; then
    curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable
    source \$HOME/.cargo/env
  fi
  source \$HOME/.cargo/env
  cd $REMOTE_DIR
  cargo build --release 2>&1
  echo '=== Build OK ==='
"

echo "=== Installation du service systemd ==="
ssh -i "$KEY" "$VPS" "
  sudo cp $REMOTE_DIR/twitninf-rust-recommender.service /etc/systemd/system/
  sudo systemctl daemon-reload
  sudo systemctl enable twitninf-rust-recommender
  sudo systemctl restart twitninf-rust-recommender
  sleep 2
  sudo systemctl status twitninf-rust-recommender --no-pager
"

echo "=== Vérification santé ==="
ssh -i "$KEY" "$VPS" "curl -s http://127.0.0.1:3002/health | python3 -m json.tool"

echo ""
echo "Déploiement terminé !"
echo "Le service écoute sur http://127.0.0.1:3002 (localhost uniquement)"
