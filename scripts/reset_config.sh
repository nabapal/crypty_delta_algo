#!/usr/bin/env bash
set -euo pipefail

# NOTE: This utility is mainly for legacy systems. As of the latest update,
# ui_overrides.json now has higher priority than ui_config_snapshot.json,
# so this manual intervention should rarely be needed.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_DIR="${REPO_ROOT}/storage/config"
SNAPSHOT_FILE="${CONFIG_DIR}/ui_config_snapshot.json"
OVERRIDES_FILE="${CONFIG_DIR}/ui_overrides.json"

echo "🔄 Config Reset Utility"
echo "======================"

if [[ -f "${SNAPSHOT_FILE}" ]]; then
    echo "📊 Current snapshot file exists:"
    echo "   Path: ${SNAPSHOT_FILE}"
    echo "   Size: $(stat -c%s "${SNAPSHOT_FILE}") bytes"
    echo ""
fi

if [[ -f "${OVERRIDES_FILE}" ]]; then
    echo "🎯 Current overrides file exists:"
    echo "   Path: ${OVERRIDES_FILE}" 
    echo "   Size: $(stat -c%s "${OVERRIDES_FILE}") bytes"
    echo ""
    
    echo "📋 Current overrides config:"
    cat "${OVERRIDES_FILE}"
    echo ""
fi

echo "Choose an option:"
echo "1. Delete snapshot (use overrides only)"
echo "2. Copy overrides to snapshot (make overrides the active config)"
echo "3. Show current config priority"
echo "4. Exit"

read -p "Enter choice (1-4): " choice

case ${choice} in
    1)
        if [[ -f "${SNAPSHOT_FILE}" ]]; then
            echo "🗑️  Deleting snapshot file..."
            rm "${SNAPSHOT_FILE}"
            echo "✅ Snapshot deleted. System will now use overrides."
        else
            echo "ℹ️  No snapshot file to delete."
        fi
        ;;
    2)
        if [[ -f "${OVERRIDES_FILE}" ]]; then
            echo "📋 Copying overrides to snapshot..."
            cp "${OVERRIDES_FILE}" "${SNAPSHOT_FILE}"
            echo "✅ Overrides copied to snapshot. Config updated."
        else
            echo "❌ No overrides file found to copy."
            exit 1
        fi
        ;;
    3)
        echo "📊 Configuration Priority Order:"
        echo "1. ui_config_snapshot.json (highest priority)"
        echo "2. ui_overrides.json (medium priority)"  
        echo "3. Built-in defaults (lowest priority)"
        echo ""
        echo "Current files:"
        if [[ -f "${SNAPSHOT_FILE}" ]]; then
            echo "✅ Snapshot exists - WILL BE USED"
        else
            echo "❌ Snapshot missing"
        fi
        if [[ -f "${OVERRIDES_FILE}" ]]; then
            echo "✅ Overrides exists - $([ -f "${SNAPSHOT_FILE}" ] && echo "will be ignored" || echo "WILL BE USED")"
        else
            echo "❌ Overrides missing"
        fi
        ;;
    4)
        echo "👋 Exiting without changes."
        exit 0
        ;;
    *)
        echo "❌ Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "🔧 Recommended next step: Restart the application to load new config"
echo "   sudo docker-compose restart"