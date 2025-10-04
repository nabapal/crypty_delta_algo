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
fi

echo "➡️  Fetching latest code..."
git fetch origin

echo "➡️  Pulling main from origin..."
git pull --ff-only origin main

echo "🛠️  Rebuilding containers..."
${COMPOSE_CMD} build

echo "♻️  Restarting services..."
${COMPOSE_CMD} down
${COMPOSE_CMD} up -d

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
