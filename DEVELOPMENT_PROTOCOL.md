# Antigravity Development Protocol

## 🚨 MANDATORY CHECKLIST 🚨
**MUST BE CHECKED AND FOLLOWED FOR EVERY TASK.**

For every feature implementation, code change, or new server addition, you **MUST** perform the following actions automatically:

### 1. 📉 Resource Management (Auto-Unload)
- [ ] **Implement Auto-Unload**: Ensure models automatically unload from VRAM after a period of inactivity (default: 60-600s depending on model size).
- [ ] **Verify Unload**: Confirm verification scripts check that VRAM is released.

### 2. 📚 Documentation Updates
- [ ] **Update README.md**: Add new services, features, or configuration changes to the main `README.md`.
- [ ] **Update API Docs**: Ensure `openapi.yaml` or equivalent API documentation reflects all endpoint changes.
- [ ] **Update Skills**: Update `SKILL.md` files in `openclaw_skills/` to match new agent capabilities.

### 3. ✅ Verification & Testing
- [ ] **Test Everything**: Run `test_all_servers.py` to ensure no regressions.
- [ ] **Verify End-to-End**: Ensure the entire flow (request -> processing -> response -> unload) works smoothly.

### 4. 🔄 Unified Launcher Sync
- [ ] **Sync Launchers**: Always update `unified_server.py` AND `run_server.bat` together when modifying server startup logic.
- [ ] **Streamlined Updates**: Ensure changes in one are reflected in the other to maintain a consistent unified experience.
- [ ] **Integrity Check**: Verify that `run_server.bat` correctly calls `unified_server.py` with necessary arguments.
