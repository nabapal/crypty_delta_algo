## Docker Configuration Fix for UI Config Persistence

### Problem
Configuration changes made through the Streamlit UI were not persisting because the config files (`ui_config_snapshot.json` and `ui_overrides.json`) were not mounted as volumes in the Docker containers.

### Solution
1. **Updated docker-compose.yml** to mount the entire `storage` directory which includes a new `config` subdirectory
2. **Modified application code** to store config files in `storage/config/` instead of the root directory
3. **Added automatic migration** of existing config files to the new location

### Changes Made

#### 1. Docker Compose Configuration
- Added `CONFIG_DIR: /app/storage/config` environment variable
- Mounted entire `./storage:/app/storage` volume for both services
- This ensures all configuration changes persist across container restarts

#### 2. Application Code Updates
- Updated `production_delta_trader.py` to use `storage/config/` for config files
- Updated `streamlit_app.py` to use `storage/config/` for config files
- Added automatic migration from old location to new location

#### 3. Directory Structure
```
storage/
├── config/
│   ├── ui_config_snapshot.json
│   └── ui_overrides.json
├── logs/
├── trades/
└── backups/
```

### Deployment Steps
1. Copy existing config files to the new location:
   ```bash
   mkdir -p storage/config
   cp ui_config_snapshot.json storage/config/
   cp ui_overrides.json storage/config/
   ```

2. Rebuild and restart the containers:
   ```bash
   docker-compose down
   docker-compose build
   docker-compose up -d
   ```

3. Verify config persistence:
   - Make changes in the UI
   - Check that files are updated in `storage/config/`
   - Restart containers and verify changes persist

### Verification
After deployment, UI configuration changes will:
- ✅ Persist across container restarts
- ✅ Be backed up automatically during redeployments
- ✅ Be stored in the persistent `storage/` directory

This fixes the issue where max profit and trailing stop-loss configurations were not being saved properly.