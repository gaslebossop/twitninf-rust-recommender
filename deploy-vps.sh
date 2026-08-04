#!/usr/bin/env bash
#
# Déploiement du recommandeur Rust sur le VPS.
#
# La version précédente de ce script ne déployait RIEN : elle compilait en local
# puis affichait les commandes à taper. Pire, compiler en local sous Windows
# produit un binaire Windows, inutilisable sur le VPS. Le seul chemin praticable
# est de compiler SUR la machine cible — c'est ce que fait ce script.
#
# Usage :
#   ./deploy-vps.sh              # envoi + compilation + redémarrage + contrôles
#   ./deploy-vps.sh --check      # contrôles seuls, ne touche à rien
#   ./deploy-vps.sh --no-restart # compile sans redémarrer (préparer une bascule)
#
set -euo pipefail

VPS_HOST="${VPS_HOST:-debian@51.210.11.74}"
VPS_KEY="${VPS_KEY:-C:\\Users\\nouno\\OneDrive\\Bureau\\Documents\\privatessh}"
REMOTE_DIR="${REMOTE_DIR:-/home/debian/rust-recommender}"

# ⚠ L'unité s'appelle `rust-recommender`, PAS `twitninf-rust-recommender`.
#
# Les deux existent sur le VPS. `twitninf-rust-recommender` est un doublon
# périmé, inactif et désactivé, dont le mot de passe base est faux : le démarrer
# produit une boucle de crash toutes les 5 s, et son échec de bind vient
# simplement de ce que la vraie unité tient déjà le port 3002. Interroger le
# mauvais nom donne l'impression que le service est éteint alors qu'il tourne.
SERVICE="${SERVICE:-rust-recommender}"
HEALTH_URL="http://127.0.0.1:3002/health"

ssh_vps() { ssh -i "$VPS_KEY" -o StrictHostKeyChecking=accept-new "$VPS_HOST" "$@"; }

log() { printf '\n\033[1;36m▶ %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# ── Contrôles d'état ─────────────────────────────────────────────────────────
#
# Toujours par le PORT et la SANTÉ, jamais par `systemctl is-active` seul :
# avec deux unités portant le même service, le nom ment.
check_health() {
  log "Contrôle de santé"
  ssh_vps "ss -tlnp 2>/dev/null | grep -q ':3002' || { echo 'rien n écoute sur 3002'; exit 1; }"

  local health
  health="$(ssh_vps "curl -s --max-time 5 '$HEALTH_URL'")" || die "/health injoignable"
  echo "$health"
  case "$health" in
    *'"db":"ok"'*) ;;
    *) die "base injoignable depuis le service" ;;
  esac
  case "$health" in
    *'"redis":"ok"'*) ;;
    *) die "redis injoignable depuis le service" ;;
  esac

  # Un service qui redémarre en boucle répond quand même à /health entre deux
  # crashs : c'est le compteur de redémarrages qui le trahit.
  local restarts
  restarts="$(ssh_vps "systemctl show '$SERVICE' -p NRestarts --value")"
  log "NRestarts = ${restarts}"
  [ "$restarts" = "0" ] || printf '\033[1;33m⚠ le service a redémarré %s fois — vérifier les journaux\033[0m\n' "$restarts"
}

if [ "${1:-}" = "--check" ]; then
  check_health
  exit 0
fi

cd "$(dirname "$0")"

# ── Tests locaux avant tout envoi ────────────────────────────────────────────
# Compiler sur le VPS coûte plus d'une minute : autant ne pas y envoyer du code
# qui ne passe pas ses propres tests.
log "Tests locaux"
cargo test --lib

# ── Envoi des sources ────────────────────────────────────────────────────────
# `target/` est exclu : le binaire local est un binaire Windows, l'envoyer
# écraserait celui du VPS par un exécutable qui ne s'y lance pas.
log "Envoi des sources vers $VPS_HOST:$REMOTE_DIR"
tar -czf - --exclude=target --exclude=.git \
    src Cargo.toml Cargo.lock \
  | ssh_vps "mkdir -p '$REMOTE_DIR' && tar -xzf - -C '$REMOTE_DIR'"

# ⚠ `tar` PRÉSERVE les dates de modification.
#
# Des sources datées d'avant le binaire font conclure à cargo que tout est à
# jour : la compilation « réussit » en 0,2 s sans rien reconstruire, le service
# redémarre sur l'ANCIEN binaire, et le déploiement paraît parfait alors que
# rien n'a changé. On repousse donc les dates après extraction.
log "Réalignement des dates de modification"
ssh_vps "cd '$REMOTE_DIR' && find src Cargo.toml Cargo.lock -type f -exec touch {} +"

# ── Compilation sur la machine cible ─────────────────────────────────────────
# cargo est installé par rustup dans ~/.cargo/bin, qui n'est PAS dans le PATH
# d'un shell non interactif : on l'appelle par son chemin complet.
log "Compilation sur le VPS (~1 min à froid)"
ssh_vps "cd '$REMOTE_DIR' && ~/.cargo/bin/cargo build --release 2>&1 | tail -20"

ssh_vps "test -x '$REMOTE_DIR/target/release/twitninf-recommender'" \
  || die "binaire absent après compilation"

if [ "${1:-}" = "--no-restart" ]; then
  log "Compilé sans redémarrage (--no-restart) — le service tourne encore sur l'ancien binaire"
  exit 0
fi

# ── Redémarrage ──────────────────────────────────────────────────────────────
# L'unité pointe déjà sur target/release/twitninf-recommender : elle prend le
# nouveau binaire sans autre manipulation. La configuration vit dans ses
# `Environment=`, il n'y a pas de .env à synchroniser.
log "Redémarrage de $SERVICE"
ssh_vps "sudo systemctl restart '$SERVICE'"

# Laisser au service le temps de se lier au port et d'ouvrir ses connexions.
sleep 5
check_health

log "Déploiement terminé"
