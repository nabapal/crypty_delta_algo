#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v git >/dev/null 2>&1; then
  echo "git is not installed or not on PATH. Aborting." >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not installed or not on PATH. Aborting." >&2
  exit 1
fi

COMPOSE_CMD="docker compose"
if ! docker compose version >/dev/null 2>&1; then
  if command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_CMD="docker-compose"
  else
    echo "Neither 'docker compose' nor 'docker-compose' is available. Aborting." >&2
    exit 1
  fi
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP_DIR="${REPO_ROOT}/storage/backups/${TIMESTAMP}"
mkdir -p "${BACKUP_DIR}"

echo "🔄 Backing up current state to ${BACKUP_DIR}..."
if [[ -f "${REPO_ROOT}/storage/config/ui_config_snapshot.json" ]]; then
  cp "${REPO_ROOT}/storage/config/ui_config_snapshot.json" "${BACKUP_DIR}/"
fi
if [[ -f "${REPO_ROOT}/storage/config/ui_overrides.json" ]]; then
  cp "${REPO_ROOT}/storage/config/ui_overrides.json" "${BACKUP_DIR}/"
fi

STASH_REF=""
if [ -n "$(git status --porcelain)" ]; then
  echo "⚠️  Detected local changes. Backing up configs to ${BACKUP_DIR} and stashing before update."
  for file in ui_config_snapshot.json ui_overrides.json; do
    if [ -f "${file}" ]; then
      cp "${file}" "${BACKUP_DIR}/${file}"
    fi
  done
  git stash push --include-untracked -m "redeploy-${TIMESTAMP}"
  STASH_REF="$(git stash list | head -n1 | cut -d: -f1)"
  echo "✅ Stashed changes as: ${STASH_REF}"
fi

echo "🔍 Current git status after stashing:"
git status --short

echo "➡️  Fetching latest code..."
if ! git fetch origin; then
  echo "❌ Failed to fetch from origin" >&2
  exit 1
fi
echo "✅ Fetch completed"

echo "➡️  Pulling main from origin..."
if ! git pull --ff-only origin main; then
  echo "❌ Failed to pull from origin (possibly merge conflicts or non-fast-forward)" >&2
  echo "Current branch status:"
  git status
  exit 1
fi
echo "✅ Pull completed"

echo "🛠️  Rebuilding containers..."
if ! ${COMPOSE_CMD} build; then
  echo "❌ Failed to build containers" >&2
  exit 1
fi
echo "✅ Build completed"

echo "♻️  Restarting services..."
echo "🛑 Stopping containers..."
if ! ${COMPOSE_CMD} down; then
  echo "❌ Failed to stop containers" >&2
  exit 1
fi
echo "✅ Containers stopped"

echo "🚀 Starting containers..."
if ! ${COMPOSE_CMD} up -d; then
  echo "❌ Failed to start containers" >&2
  exit 1
fi
echo "✅ Containers started"

if [ -n "${STASH_REF}" ]; then
  echo "🔁 Attempting to reapply stashed changes (${STASH_REF})."
  if git stash pop "${STASH_REF}"; then
    echo "✅ Local changes restored from stash."
  else
    echo "⚠️  Could not cleanly reapply stashed changes. Restoring backups instead."
    git reset --hard HEAD
    for file in ui_config_snapshot.json ui_overrides.json; do
      if [ -f "${BACKUP_DIR}/${file}" ]; then
        cp "${BACKUP_DIR}/${file}" "${file}"
      fi
    done
    echo "Manual review may be required. Your original files are also saved in ${BACKUP_DIR}."
  fi
fi

echo "✅ Redeploy complete. Active containers:"
${COMPOSE_CMD} ps
