<#
.SYNOPSIS
  Installation d'Aria sous Windows : prérequis (optionnels), build de l'image et création du conteneur.

.DESCRIPTION
  Rien n'est installé sur le système sans -InstallDeps. La clé API ne vit que dans .env, que ce script ne lit jamais.
  Sous Windows, Podman et Docker passent par WSL2 : le GPU NVIDIA est visible dans WSL2 avec le pilote Windows NVIDIA
  (aucun pilote à installer dans WSL). Le NVIDIA Container Toolkit se configure à part, dans la machine WSL.

.EXAMPLE
  .\scripts\install.ps1
  Build CPU + démarrage (Podman ou Docker, détecté).

.EXAMPLE
  .\scripts\install.ps1 -Gpu -LlamaCuda -CudaArch 89

.EXAMPLE
  .\scripts\install.ps1 -DryRun -Gpu -LlamaCuda
  Affiche ce qui serait fait, sans rien exécuter.
#>
[CmdletBinding()]
param(
  [ValidateSet('', 'podman', 'docker')][string]$Engine = '',   # vide = détection (podman d'abord)
  [switch]$Gpu,            # donne le GPU NVIDIA au conteneur
  [switch]$LlamaCuda,      # compile llama-cpp-python avec CUDA (LLM local sur GPU)
  [string]$CudaArch = '',  # architecture CUDA ciblée, ex. 89 ; vide = toutes (compilation bien plus longue)
  [switch]$FlashAttn,      # compile flash-attn (GPU de capacité 8.0+, compilation très longue)
  [switch]$NoLock,         # requirements.txt au lieu de requirements.lock
  [switch]$InstallDeps,    # installe les prérequis manquants (via winget)
  [switch]$NoStart,        # construit l'image sans démarrer le conteneur
  [switch]$NoCache,        # build sans cache
  [switch]$DryRun          # affiche les commandes sans les exécuter
)

$ErrorActionPreference = 'Stop'

function Log($m)  { Write-Host "[aria] $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "[aria] ATTENTION : $m" -ForegroundColor Yellow }
function Die($m)  { Write-Host "[aria] ERREUR : $m" -ForegroundColor Red; exit 1 }
function Have($n) { [bool](Get-Command $n -ErrorAction SilentlyContinue) }
# Sonde : exécute une commande native en ignorant sa sortie ; renvoie son code de retour.
# (Windows PowerShell 5.1 transforme tout message sur stderr en erreur fatale avec 'Stop', même redirigé vers $null.)
function Probe {
  param([string]$Exe, [string[]]$Arguments = @())
  $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
  try { & $Exe @Arguments 2>&1 | Out-Null; return $LASTEXITCODE } finally { $ErrorActionPreference = $old }
}
function Run {
  param([string]$Exe, [string[]]$Arguments = @())
  Write-Host ("  > {0} {1}" -f $Exe, ($Arguments -join ' '))
  if (-not $DryRun) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { Die "La commande a échoué (code $LASTEXITCODE) : $Exe $($Arguments -join ' ')" }
  }
}

Set-Location (Split-Path -Parent $PSScriptRoot)

# --- Validation des paramètres ---------------------------------------------------------------------------------
if ($CudaArch -ne '' -and $CudaArch -notmatch '^\d{2,3}([;,]\d{2,3})*$') {
  Die "-CudaArch invalide : '$CudaArch' (attendu : 89, ou plusieurs séparés par ; ou , )"
}
$CudaArch = $CudaArch -replace ',', ';'
if ($LlamaCuda -and -not $Gpu) { Warn "-LlamaCuda sans -Gpu : llama-cpp sera compilé avec CUDA, mais le conteneur n'aura pas accès au GPU à l'exécution." }
if ($CudaArch -ne '' -and -not $LlamaCuda) { Warn "-CudaArch ignoré sans -LlamaCuda." }

# --- Moteur de conteneurs -----------------------------------------------------------------------------------------
if ($Engine -eq '') {
  if (Have 'podman') { $Engine = 'podman' } elseif (Have 'docker') { $Engine = 'docker' }
}
if ($Engine -eq '' -or -not (Have $Engine)) {
  $want = if ($Engine -ne '') { $Engine } else { 'podman' }
  if (-not $InstallDeps) { Die "Aucun moteur de conteneurs trouvé. Installez Podman ou Docker, ou relancez avec -InstallDeps." }
  if (-not (Have 'winget')) { Die "winget est introuvable : installez $want à la main (https://podman.io ou https://docker.com)." }
  $id = if ($want -eq 'podman') { 'RedHat.Podman' } else { 'Docker.DockerDesktop' }
  Log "Installation de $want via winget ($id)..."
  Run 'winget' @('install', '--id', $id, '-e', '--accept-package-agreements', '--accept-source-agreements')
  Warn "Après l'installation : ouvrez un NOUVEAU terminal ($want doit être dans le PATH)."
  if ($want -eq 'podman') {
    Log "Initialisation de la machine Podman (WSL2)..."
    Run 'podman' @('machine', 'init', '--cpus', '4', '--memory', '8192', '--disk-size', '60')
    Run 'podman' @('machine', 'start')
  }
  if (-not $DryRun) { Warn "Relancez le script dans un nouveau terminal."; exit 0 }
  $Engine = $want
}
Log "Moteur de conteneurs : $Engine"

if ($Engine -eq 'podman' -and -not $DryRun) {
  $machines = podman machine list --format '{{.Name}}|{{.Running}}' 2>$null
  if (-not $machines)                   { Die "Aucune machine Podman. Lancez : podman machine init   puis   podman machine start" }
  if ($machines -notmatch '\|true')     { Log "Démarrage de la machine Podman..."; Run 'podman' @('machine', 'start') }
}

# --- Commande « compose » ------------------------------------------------------------------------------------------
$compose = $null
if ($Engine -eq 'docker') {
  if ((Probe 'docker' @('compose', 'version')) -eq 0) { $compose = @('docker', 'compose') } elseif (Have 'docker-compose') { $compose = @('docker-compose') }
} else {
  if ((Probe 'podman' @('compose', 'version')) -eq 0) { $compose = @('podman', 'compose') } elseif (Have 'podman-compose') { $compose = @('podman-compose') }
}
if (-not $compose) {
  if (-not $InstallDeps) { Die "Aucun « compose » pour $Engine. Installez-le (docker compose / pip install podman-compose) ou relancez avec -InstallDeps." }
  if ($Engine -eq 'podman') {
    if (-not (Have 'pip')) { Die "pip introuvable : installez Python, puis : pip install podman-compose" }
    Run 'pip' @('install', 'podman-compose')
    $compose = @('podman-compose')
  } else { Die "Docker Desktop fournit « docker compose » : réinstallez-le ou mettez-le à jour." }
}
Log "Compose : $($compose -join ' ')"

# --- GPU -------------------------------------------------------------------------------------------------------------------
$files = @('-f', 'docker-compose.yml')
if ($Gpu) {
  if (-not (Have 'nvidia-smi')) { Warn "nvidia-smi introuvable : pilote NVIDIA absent de Windows ? Le conteneur ne verra pas de GPU." }
  if ($Engine -eq 'docker') {
    Log "Docker Desktop (WSL2) expose le GPU directement : rien à installer."
    $files += @('-f', 'docker-compose.gpu-docker.yml')
  } else {
    $hasCtk = $false
    if (-not $DryRun) { $hasCtk = ((Probe 'podman' @('machine', 'ssh', 'command -v nvidia-ctk')) -eq 0) }
    if (-not $hasCtk) {
      if (-not $InstallDeps) {
        Die "NVIDIA Container Toolkit absent de la machine Podman. Relancez avec -InstallDeps, ou voir https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
      }
      Log "Installation du NVIDIA Container Toolkit dans la machine Podman (Fedora : dnf)..."
      Run 'podman' @('machine', 'ssh', 'curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo > /dev/null && sudo dnf install -y nvidia-container-toolkit')
    }
    Log "Génération de la spécification CDI dans la machine Podman..."
    Run 'podman' @('machine', 'ssh', 'sudo mkdir -p /etc/cdi && sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml')
    $files += @('-f', 'docker-compose.gpu-podman.yml')
  }
}

# --- Fichier .env : créé, jamais lu ------------------------------------------------------------------------------------
if (-not (Test-Path '.env')) {
  Log "Création de .env (renseignez-y ANTHROPIC_API_KEY si vous utilisez Claude)."
  if (Test-Path '.env.example') { if (-not $DryRun) { Copy-Item '.env.example' '.env' } else { Write-Host "  > Copy-Item .env.example .env" } }
  else { if (-not $DryRun) { New-Item -ItemType File -Path '.env' | Out-Null } else { Write-Host "  > New-Item .env" } }
}

# --- Build -----------------------------------------------------------------------------------------------------------------------
$env:LLAMA_CUDA       = if ($LlamaCuda) { 'true' } else { 'false' }
$env:CUDA_ARCH        = $CudaArch
$env:BUILD_FLASH_ATTN = if ($FlashAttn) { 'true' } else { 'false' }
$env:USE_REQ_LOCK     = if ($NoLock) { 'false' } else { 'true' }
Log ("Paramètres : llama-cuda={0} arch={1} flash-attn={2} lock={3} gpu={4}" -f $env:LLAMA_CUDA, $(if ($CudaArch) { $CudaArch } else { 'toutes' }), $env:BUILD_FLASH_ATTN, $env:USE_REQ_LOCK, [bool]$Gpu)
if ($LlamaCuda -and $CudaArch -eq '') { Warn "Sans -CudaArch, llama-cpp est compilé pour toutes les architectures : très long. Ex. : -CudaArch 89" }

$buildArgs = @('build'); if ($NoCache) { $buildArgs += '--no-cache' }
Log "Construction de l'image (la première fois : plusieurs minutes, plusieurs Go à télécharger)..."
$cmd = $compose[0]; $pre = @(); if ($compose.Count -gt 1) { $pre = $compose[1..($compose.Count - 1)] }
Run $cmd ($pre + $files + $buildArgs)

if (-not $NoStart) {
  # Podman ne crée pas les dossiers montés (Docker, si) : sans lui, « statfs ... models: no such file »
  if (-not $DryRun -and -not (Test-Path 'models')) { New-Item -ItemType Directory -Path 'models' | Out-Null }
  Log "Démarrage du conteneur..."
  Run $cmd ($pre + $files + @('up', '-d'))
  Log "Aria démarre (20 à 60 s) : http://localhost:7860"
  Log "Logs : $Engine logs -f aria"
} else {
  Log "Image construite. Démarrage : $($compose -join ' ') $($files -join ' ') up -d"
}
