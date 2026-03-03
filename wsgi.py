# ============================================================================
# wsgi.py — Production entry point for Gunicorn / Render
# ============================================================================
# Usage:  gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers 1 --threads 4
# ============================================================================

import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the Flask app object
from A1_Boot_Dash import (
    app, init_db, import_csv_to_db,
    camera_loop, poll_esp32_device, auto_light_controller, lights_automation_loop
)
from threading import Thread

# ---------- One-time initialization (runs once per worker) ----------
_initialized = False

def _startup():
    global _initialized
    if _initialized:
        return
    _initialized = True

    print("[WSGI] Running startup initialization")
    init_db()
    import_csv_to_db()

    # Start background threads (they will fail gracefully if ESP32 is unreachable)
    for target, name in [
        (camera_loop,            "Camera"),
        (poll_esp32_device,      "ESP32 polling"),
        (auto_light_controller,  "Auto-light"),
        (lights_automation_loop, "Schedule fallback"),
    ]:
        t = Thread(target=target, daemon=True)
        t.start()
        print(f"[WSGI] {name} thread started")

    print("[WSGI] Startup complete")

_startup()
