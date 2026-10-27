#!/usr/bin/env bash
# Installation d'Aria sous Linux : prérequis (optionnels), build de l'image et création du conteneur.
#
# Exemples :
#   scripts/install.sh                                  # build CPU + démarrage (podman ou docker, détecté)
#   scripts/install.sh --gpu --llama-cuda --cuda-arch 89
#   scripts/install.sh --engine docker --gpu --install-deps
#   scripts/install.sh --dry-run --gpu --llama-cuda     # affiche ce qui serait fait, sans rien exécuter
#
# Rien n'est installé sur le système sans --install-deps. La clé API ne vit que dans .env, que ce script ne lit jamais.
set -euo pipefail

ENGINE=""            # podman | docker (vide = détection automatique)
GPU=false            # donne le GPU NVIDIA au conteneur
LLAMA_CUDA=false     # compile llama-cpp-python avec CUDA (LLM local sur GPU)
CUDA_ARCH=""         # architecture CUDA ciblée, ex. 89 ; vide = toutes (compilation bien plus longue)
FLASH_ATTN=false     # compile flash-attn (GPU de capacité 8.0+, compilation très longue)
USE_LOCK=true        # installer les dépendances depuis requirements.lock (false = requirements.txt)
INSTALL_DEPS=false   # installe les prérequis manquants (moteur de conteneurs, NVIDIA Container Toolkit)
START=true           # démarre le conteneur après le build
NO_CACHE=false       # build sans cache
DRY_RUN=false        # affiche les commandes sans les exécuter

usage() { sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; cat <<'EOF'

Options :
  --engine podman|docker   moteur de conteneurs (défaut : podman s'il est installé, sinon docker)
  --gpu                    donne le GPU NVIDIA au conteneur
  --llama-cuda             compile llama-cpp-python avec CUDA
  --cuda-arch N            architecture CUDA (ex. 89 pour RTX Ada, 86 pour RTX 30xx, 75 pour RTX 20xx)
  --flash-attn             compile flash-attn
  --no-lock                utilise requirements.txt au lieu de requirements.lock
  --install-deps           installe les prérequis manquants (moteur, compose, NVIDIA Container Toolkit) via sudo
  --no-start               construit l'image sans démarrer le conteneur
  --no-cache               build sans cache
  --dry-run                affiche les commandes sans les exécuter
  -h, --help               cette aide
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --engine)       ENGINE="${2:-}"; shift 2 ;;
    --gpu)          GPU=true; shift ;;
    --llama-cuda)   LLAMA_CUDA=true; shift ;;
    --cuda-arch)    CUDA_ARCH="${2:-}"; shift 2 ;;
    --flash-attn)   FLASH_ATTN=true; shift ;;
    --no-lock)      USE_LOCK=false; shift ;;
    --install-deps) INSTALL_DEPS=true; shift ;;
    --no-start)     START=false; shift ;;
    --no-cache)     NO_CACHE=true; shift ;;
    --dry-run)      DRY_RUN=true; shift ;;
    -h|--help)      usage; exit 0 ;;
    *) echo "Option inconnue : $1" >&2; usage >&2; exit 2 ;;
  esac
done

log()  { printf '\033[1;34m[aria]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[aria] ATTENTION :\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[aria] ERREUR :\033[0m %s\n' "$*" >&2; exit 1; }
run()  { printf '  $ %s\n' "$*"; $DRY_RUN || "$@"; }
have() { command -v "$1" >/dev/null 2>&1; }

cd "$(dirname "$0")/.."

# Ce script cible Linux. Sous Windows (Git Bash, MSYS, Cygwin), --dry-run reste autorisé pour vérifier les paramètres.
case "$(uname -s)" in
  Linux*) ;;
  *) $DRY_RUN || die "Ce script est prévu pour Linux. Sous Windows, utilisez : .\\scripts\\install.ps1" ;;
esac

# --- Validation des paramètres ------------------------------------------------------------------------------
case "$ENGINE" in ""|podman|docker) ;; *) die "--engine doit valoir podman ou docker (reçu : '$ENGINE')" ;; esac
if [ -n "$CUDA_ARCH" ] && ! [[ "$CUDA_ARCH" =~ ^[0-9]{2,3}([;,][0-9]{2,3})*$ ]]; then
  die "--cuda-arch invalide : '$CUDA_ARCH' (attendu : 89, ou plusieurs séparés par ; ou , )"
fi
CUDA_ARCH="${CUDA_ARCH//,/;}"
[ "$LLAMA_CUDA" = true ] && [ "$GPU" = false ] && \
  warn "--llama-cuda sans --gpu : llama-cpp sera compilé avec CUDA, mais le conteneur n'aura pas accès au GPU à l'exécution."
[ -n "$CUDA_ARCH" ] && [ "$LLAMA_CUDA" = false ] && warn "--cuda-arch ignoré sans --llama-cuda."

# --- Moteur de conteneurs -------------------------------------------------------------------------------------
detect_pkg_manager() {
  for m in apt-get dnf zypper pacman; do have "$m" && { echo "$m"; return; }; done
  echo ""
}
PKG="$(detect_pkg_manager)"

sudo_run() {
  if [ "$(id -u)" -eq 0 ]; then run "$@"; else have sudo || die "sudo est nécessaire pour : $*"; run sudo "$@"; fi
}

install_engine() {
  local want="${ENGINE:-podman}"
  log "Installation de $want via ${PKG:-?}..."
  case "$PKG" in
    apt-get) sudo_run apt-get update; sudo_run apt-get install -y "$want" ;;
    dnf)     sudo_run dnf install -y "$want" ;;
    zypper)  sudo_run zypper install -y "$want" ;;
    pacman)  sudo_run pacman -S --noconfirm "$want" ;;
    *) die "Gestionnaire de paquets non reconnu : installez $want à la main." ;;
  esac
  ENGINE="$want"
}

if [ -z "$ENGINE" ]; then
  if have podman; then ENGINE=podman; elif have docker; then ENGINE=docker; fi
fi
if [ -z "$ENGINE" ] || ! have "$ENGINE"; then
  if [ "$INSTALL_DEPS" = true ]; then install_engine
  else die "Aucun moteur de conteneurs trouvé. Installez podman ou docker, ou relancez avec --install-deps."; fi
fi
log "Moteur de conteneurs : $ENGINE"

# --- Commande « compose » ---------------------------------------------------------------------------------------
if   [ "$ENGINE" = docker ] && docker compose version >/dev/null 2>&1; then COMPOSE=(docker compose)
elif [ "$ENGINE" = podman ] && podman compose version >/dev/null 2>&1;  then COMPOSE=(podman compose)
elif have docker-compose && [ "$ENGINE" = docker ];                    then COMPOSE=(docker-compose)
elif have podman-compose && [ "$ENGINE" = podman ];                    then COMPOSE=(podman-compose)
else
  if [ "$INSTALL_DEPS" = true ]; then
    case "$ENGINE:$PKG" in
      docker:apt-get) sudo_run apt-get install -y docker-compose-v2 ;;
      docker:dnf)     sudo_run dnf install -y docker-compose-plugin ;;
      podman:apt-get) sudo_run apt-get install -y podman-compose ;;
      podman:dnf)     sudo_run dnf install -y podman-compose ;;
      *) die "Installez le plugin compose de $ENGINE à la main." ;;
    esac
    COMPOSE=("$ENGINE" compose)
  else
    die "Aucun « compose » pour $ENGINE. Installez-le (docker compose, podman-compose) ou relancez avec --install-deps."
  fi
fi
log "Compose : ${COMPOSE[*]}"

# --- GPU : prérequis NVIDIA -----------------------------------------------------------------------------------------
COMPOSE_FILES=(-f docker-compose.yml)
if [ "$GPU" = true ]; then
  have nvidia-smi || warn "nvidia-smi introuvable : pilote NVIDIA absent de l'hôte ? Le conteneur ne verra pas de GPU."
  if ! have nvidia-ctk; then
    if [ "$INSTALL_DEPS" = true ]; then
      log "Installation du NVIDIA Container Toolkit (nécessaire pour donner le GPU aux conteneurs)..."
      case "$PKG" in
        apt-get)
          sudo_run apt-get install -y --no-install-recommends ca-certificates curl gnupg2
          if ! $DRY_RUN; then
            curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
            curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
              | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
              | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
          else echo "  (ajout du dépôt NVIDIA apt)"; fi
          sudo_run apt-get update
          sudo_run apt-get install -y nvidia-container-toolkit ;;
        dnf)
          if ! $DRY_RUN; then
            curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
              | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo >/dev/null
          else echo "  (ajout du dépôt NVIDIA rpm)"; fi
          sudo_run dnf install -y nvidia-container-toolkit ;;
        *) die "Installez le NVIDIA Container Toolkit à la main : aucun gestionnaire de paquets géré (apt-get, dnf) n'a été trouvé sur ce système." ;;
      esac
    else
      die "NVIDIA Container Toolkit absent. Relancez avec --install-deps, ou installez-le : https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
    fi
  fi
  if [ "$ENGINE" = docker ]; then
    sudo_run nvidia-ctk runtime configure --runtime=docker
    if $DRY_RUN; then echo "  \$ sudo systemctl restart docker"
    elif have systemctl; then sudo systemctl restart docker
    else warn "Redémarrez le démon docker à la main pour prendre en compte le GPU."; fi
    COMPOSE_FILES+=(-f docker-compose.gpu-docker.yml)
  else
    if [ ! -s /etc/cdi/nvidia.yaml ] && [ ! -s /var/run/cdi/nvidia.yaml ]; then
      log "Génération de la spécification CDI pour Podman..."
      sudo_run mkdir -p /etc/cdi
      sudo_run nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
    else
      log "Spécification CDI déjà présente."
    fi
    COMPOSE_FILES+=(-f docker-compose.gpu-podman.yml)
  fi
fi

# --- Fichier .env : créé vide, jamais lu --------------------------------------------------------------------------------
if [ ! -f .env ]; then
  log "Création de .env à partir de .env.example (renseignez-y ANTHROPIC_API_KEY si vous utilisez Claude)."
  if [ -f .env.example ]; then run cp .env.example .env; else run touch .env; fi
fi

# --- Build -------------------------------------------------------------------------------------------------------------------------
export LLAMA_CUDA CUDA_ARCH
export BUILD_FLASH_ATTN="$FLASH_ATTN" USE_REQ_LOCK="$USE_LOCK"
log "Paramètres : llama-cuda=$LLAMA_CUDA arch=${CUDA_ARCH:-toutes} flash-attn=$FLASH_ATTN lock=$USE_LOCK gpu=$GPU"
[ "$LLAMA_CUDA" = true ] && [ -z "$CUDA_ARCH" ] && \
  warn "Sans --cuda-arch, llama-cpp est compilé pour toutes les architectures : très long. Ex. : --cuda-arch 89"

BUILD_ARGS=(build); $NO_CACHE && BUILD_ARGS+=(--no-cache)
log "Construction de l'image (la première fois : plusieurs minutes, plusieurs Go à télécharger)..."
run "${COMPOSE[@]}" "${COMPOSE_FILES[@]}" "${BUILD_ARGS[@]}"

if [ "$START" = true ]; then
  # Podman ne crée pas les dossiers montés (Docker, si)
  [ "$DRY_RUN" = true ] || mkdir -p models
  log "Démarrage du conteneur..."
  run "${COMPOSE[@]}" "${COMPOSE_FILES[@]}" up -d
  log "Aria démarre (20 à 60 s) : http://localhost:7860"
  log "Logs : $ENGINE logs -f aria"
else
  log "Image construite. Démarrage : ${COMPOSE[*]} ${COMPOSE_FILES[*]} up -d"
fi
