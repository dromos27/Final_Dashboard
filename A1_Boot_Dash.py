# ============================================================================
# A1_Boot_Dash.py — COMPLETE FILE (v3.5 — energy_data.csv integration)
# ============================================================================

from flask import Flask, render_template, send_from_directory, jsonify, request, Response
import os
import time
import numpy as np
import sqlite3
import csv
import io
import re
from datetime import datetime, date, timedelta
from threading import Thread, Lock
from collections import defaultdict
import sys
import requests

# Cloud mode: disables camera/ESP32/YOLO to fit in 512 MB RAM (Render free tier)
CLOUD_MODE = os.environ.get('CLOUD_MODE', '').lower() in ('1', 'true', 'yes')

if not CLOUD_MODE:
    try:
        import cv2
        import torch
    except ImportError:
        cv2 = None
        torch = None
        CLOUD_MODE = True
        print('cv2/torch not installed — forcing CLOUD_MODE')
else:
    cv2 = None
    torch = None
    print('CLOUD_MODE enabled — camera/YOLO disabled')

# ============================================================================
# PROJECT SETUP
# ============================================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

app = Flask(__name__)

# ============================================================================
# ESP32 CONFIGURATION
# ============================================================================

ESP32_CAM_IP         = "172.20.10.5"
ESP32_CAM_STREAM_URL = f"http://{ESP32_CAM_IP}/stream"
ESP32_DEVICE_IP      = "172.20.10.4"
ESP32_DEVICE_BASE_URL = f"http://{ESP32_DEVICE_IP}/"
DETECTION_COOLDOWN   = 2


# ============================================================================
# YOLO MODEL (skipped in cloud mode)
# ============================================================================

yolo_model = None
device = "cpu"

if not CLOUD_MODE:
    device = "cuda" if torch and torch.cuda.is_available() else "cpu"
    print(f"[✓] Using device: {device}")
    try:
        from ultralytics import YOLO
        yolo_model = YOLO("yolov8n.pt").float()
        yolo_model.fuse()
        if device == "cuda":
            yolo_model = yolo_model.half()
        yolo_model.to(device)
        yolo_model.conf = 0.5
        print("[✓] YOLO model loaded successfully")
    except Exception as e:
        print(f"[!] YOLO model load failed: {e}")
else:
    print("[✓] CLOUD_MODE — YOLO model skipped")

# ============================================================================
# DATABASE SETUP
# ============================================================================

DB_PATH = os.path.join(current_dir, "energy_dashboard.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS energy_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            room TEXT DEFAULT '705',
            voltage REAL DEFAULT 0,
            current REAL DEFAULT 0,
            power REAL DEFAULT 0,
            energy_kwh REAL DEFAULT 0,
            light_state TEXT DEFAULT 'OFF'
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS person_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            event TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS device_status (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            wifi_status TEXT DEFAULT 'Unknown',
            cloud_status TEXT DEFAULT 'Unknown',
            sd_card_status TEXT DEFAULT 'Unknown'
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_energy_ts ON energy_log(timestamp)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_person_ts ON person_events(timestamp)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_device_ts  ON device_status(timestamp)")
    conn.commit()
    conn.close()
    print(f"[✓] Database initialized: {DB_PATH}")


def db_log_energy(room, voltage, current, power, energy_kwh, light_state="OFF"):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO energy_log (timestamp,room,voltage,current,power,energy_kwh,light_state) VALUES (?,?,?,?,?,?,?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             room, round(voltage, 4), round(current, 6), round(power, 4), round(energy_kwh, 8), light_state)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB] Error logging energy: {e}")


def db_log_device_status(wifi, cloud, sd_card):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO device_status (timestamp,wifi_status,cloud_status,sd_card_status) VALUES (?,?,?,?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), wifi, cloud, sd_card)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB] Error logging status: {e}")


def db_get_latest_reading(room="705"):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM energy_log WHERE room=? ORDER BY timestamp DESC LIMIT 1", (room,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        print(f"[DB] Error getting latest reading: {e}")
        return None


def db_get_hourly_data(room="705", hours=12):
    try:
        since = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
        conn  = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows  = conn.execute(
            """SELECT strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
                      AVG(voltage) as avg_voltage, AVG(current) as avg_current,
                      AVG(power) as avg_power,
                      MAX(energy_kwh) - MIN(energy_kwh) as energy_consumed
               FROM energy_log WHERE room=? AND timestamp >= ?
               GROUP BY hour ORDER BY hour ASC""",
            (room, since)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"[DB] Error getting hourly data: {e}")
        return []


def db_export_to_csv(room="705", days=7):
    try:
        since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        conn  = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows  = conn.execute(
            "SELECT * FROM energy_log WHERE room=? AND timestamp >= ? ORDER BY timestamp ASC",
            (room, since)
        ).fetchall()
        conn.close()
        if not rows:
            return None
        csv_filename = f"energy_export_room{room}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_path = os.path.join(current_dir, 'exports', csv_filename)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp','room','voltage','current','power','energy_kwh','light_state'])
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
        print(f"[✓] CSV exported: {csv_path}")
        return csv_filename
    except Exception as e:
        print(f"[DB] Error exporting CSV: {e}")
        return None


def db_prune_old_data(keep_days=365):
    try:
        cutoff = (datetime.now() - timedelta(days=keep_days)).strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM energy_log   WHERE timestamp < ?", (cutoff,))
        conn.execute("DELETE FROM person_events WHERE timestamp < ?", (cutoff,))
        conn.execute("DELETE FROM device_status WHERE timestamp < ?", (cutoff,))
        conn.commit()
        conn.close()
        print(f"[✓] Pruned data older than {keep_days} days")
    except Exception as e:
        print(f"[DB] Prune error: {e}")


def import_csv_to_db():
    """
    One-time migration: reads assets/energy_data.csv and inserts any rows
    that don't already exist in energy_log. Safe to call every startup.
    Uses pandas to_sql for fast bulk insert (~seconds for 400k rows).
    """
    import pandas as pd

    csv_path = os.path.join(current_dir, 'assets', 'energy_data.csv')
    if not os.path.exists(csv_path):
        print("[CSV→DB] No energy_data.csv found — skipping import")
        return

    print(f"[CSV→DB] Starting import from {csv_path}…")
    try:
        # ── Parse CSV ────────────────────────────────────────────────────────
        df = pd.read_csv(csv_path, encoding='utf-8-sig', dtype=str)
        df.columns = [c.strip().lstrip('\ufeff') for c in df.columns]
        print(f"[CSV→DB] Read {len(df)} rows, columns: {list(df.columns)}")

        print(f"[CSV→DB] Date sample: {df['Date'].head(5).tolist()}")

        # Drop rows with empty/null Date or Time
        df = df.dropna(subset=['Date', 'Time'])
        df = df[df['Date'].str.strip() != '']
        df = df[df['Time'].str.strip() != '']
        print(f"[CSV→DB] Rows with valid Date+Time: {len(df)}")

        # Parse datetime — Date is YYYY-MM-DD, Time is h:MM:SS AM/PM
        dt_str = df['Date'].str.strip() + ' ' + df['Time'].str.strip()
        df['_dt'] = pd.to_datetime(dt_str, format='%Y-%m-%d %I:%M:%S %p', errors='coerce')
        # Fallback for 24-hour time format
        mask = df['_dt'].isna()
        if mask.any():
            df.loc[mask, '_dt'] = pd.to_datetime(
                dt_str[mask], format='%Y-%m-%d %H:%M:%S', errors='coerce'
            )
        nat_count = df['_dt'].isna().sum()
        print(f"[CSV→DB] Datetime parse: {len(df) - nat_count} ok, {nat_count} failed")
        df = df.dropna(subset=['_dt'])

        # Filter out garbage years — only keep realistic dates (2020–2030)
        df = df[df['_dt'].dt.year.between(2020, 2030)].copy()
        df = df.dropna(subset=['_dt'])
        df = df[df['_dt'].dt.year.between(2000, 2099)].copy()

        # Numeric columns
        for col in ['Voltage(V)', 'Current(A)', 'Power(W)', 'Energy(kWh)']:
            df[col] = pd.to_numeric(df.get(col, 0), errors='coerce').fillna(0)

        # Build final DataFrame matching energy_log schema
        import_df = pd.DataFrame({
            'timestamp':   df['_dt'].dt.strftime('%Y-%m-%d %H:%M:%S'),
            'room':        '705',
            'voltage':     df['Voltage(V)'].round(4),
            'current':     df['Current(A)'].round(6),
            'power':       df['Power(W)'].round(4),
            'energy_kwh':  df['Energy(kWh)'].round(8),
            'light_state': 'OFF',
        })
        print(f"[CSV→DB] Parsed {len(import_df)} valid rows spanning "
              f"{import_df['timestamp'].min()} → {import_df['timestamp'].max()}")

        # ── Insert into DB ───────────────────────────────────────────────────
        conn = sqlite3.connect(DB_PATH)

        # Purge any previously imported rows with bad years
        purged = conn.execute(
            "DELETE FROM energy_log WHERE room='705' AND (CAST(strftime('%Y', timestamp) AS INTEGER) < 2020 OR CAST(strftime('%Y', timestamp) AS INTEGER) > 2030)"
        ).rowcount
        if purged:
            print(f"[CSV→DB] Purged {purged} rows with bad years from DB")
        conn.commit()

        before = conn.execute("SELECT COUNT(*) FROM energy_log WHERE room='705'").fetchone()[0]
        print(f"[CSV→DB] DB currently has {before} rows for room 705")

        # Ensure unique index exists so INSERT OR IGNORE works
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_energy_ts_room
            ON energy_log(timestamp, room)
        """)
        conn.commit()

        # Write CSV rows to a temp table, then INSERT OR IGNORE into energy_log
        import_df.to_sql('_csv_import_tmp', conn, if_exists='replace', index=False)
        conn.execute("""
            INSERT OR IGNORE INTO energy_log
                (timestamp, room, voltage, current, power, energy_kwh, light_state)
            SELECT timestamp, room, voltage, current, power, energy_kwh, light_state
            FROM _csv_import_tmp
        """)
        conn.execute("DROP TABLE IF EXISTS _csv_import_tmp")
        conn.commit()

        after = conn.execute("SELECT COUNT(*) FROM energy_log WHERE room='705'").fetchone()[0]
        conn.close()

        added = after - before
        print(f"[CSV→DB] ✅ Import done — {added} new rows added ({after} total in DB)")

    except Exception as e:
        print(f"[CSV→DB] ❌ Error: {e}")
        import traceback; traceback.print_exc()



# ============================================================================
# SCHEDULE LOADING
# ============================================================================

def load_schedule_by_day(room, day_name):
    """Load schedule from assets/room_schedules.csv"""
    try:
        schedule_file = os.path.join(current_dir, 'assets', 'room_schedules.csv')

        if not os.path.exists(schedule_file):
            print(f"[SCHEDULE] ❌ File not found: {schedule_file}")
            return []

        schedule_data = []
        with open(schedule_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_room = str(row.get('room', '')).strip()
                row_day  = str(row.get('day',  '')).strip().lower()
                if row_room == str(room).strip() and row_day == day_name.strip().lower():
                    schedule_data.append({
                        'start_time': str(row.get('start_time', '')).strip(),
                        'end_time':   str(row.get('end_time',   '')).strip(),
                        'event_name': str(row.get('event_name', '')).strip(),
                        'day':        row_day
                    })

        return schedule_data

    except Exception as e:
        print(f"[SCHEDULE] ❌ Error: {e}")
        import traceback; traceback.print_exc()
        return []


# ============================================================================
# SHARED STATE
# ============================================================================

frame_lock   = Lock()
shared_frame = None
person_count_global    = 0
avg_confidence_global  = 0
camera_connected       = False
last_notification_time = 0

esp32_status = {
    "wifi": "Unknown", "cloud": "Unknown", "sd_card": "Unknown",
    "voltage": 0.0, "current": 0.0, "power": 0.0, "energy_kwh": 0.0,
    "light_state": "OFF", "ph_time": "--:--:--", "uptime": 0,
    "last_update": "Never", "error": None
}
esp32_lock = Lock()

# Rolling history arrays (last 60 readings) for sparkline charts
power_history   = []
voltage_history = []
current_history = []
MAX_HISTORY     = 60

detection_enabled = True
PERSON_CONFIDENCE_THRESHOLD = 0.70   # detections below this % are ignored (not counted, don't trigger lights)

# ============================================================================
# ESP32 POLLING THREAD
# ============================================================================

def parse_esp32_response(text: str) -> dict:
    """
    Parse the plain-text /info response from the ESP32.
    Extracts all numeric fields, status flags, and the ESP32's own NTP timestamp.
    """
    data = {}
    patterns = {
        "voltage":    r"Voltage:\s*([\d.]+)\s*V",
        "current":    r"Current:\s*([\d.]+)\s*A",
        "power":      r"Power:\s*([\d.]+)\s*W",
        "energy_kwh": r"Energy:\s*([\d.]+)\s*kWh",
        "uptime":     r"Uptime:\s*(\d+)\s*seconds",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            data[key] = float(m.group(1))

    data["light_state"] = "ON"  if "Light: ON"          in text else "OFF"
    data["wifi"]        = "Connected"    if "WiFi: Connected"  in text else "Disconnected"
    data["cloud"]       = "Connected"    if "Cloud: Connected" in text else "Disconnected"
    data["sd_card"]     = "OK"    if "SD Card: OK"     in text else \
                          "Failed" if "SD Card: Failed" in text else "Unknown"

    # Pull the ESP32's own NTP-synced timestamp — format: "Time: MM/DD/YYYY HH:MM:SS"
    m = re.search(r"Time:\s*(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})", text)
    if m:
        data["ph_time"] = m.group(1)

    return data


def _esp32_dt(parsed: dict) -> datetime:
    """
    Return a datetime from the ESP32's NTP-synced ph_time field.
    Falls back to datetime.now() if ph_time is absent or unparseable.
    """
    ph = parsed.get("ph_time", "")
    if ph:
        for fmt in ("%m/%d/%Y %H:%M:%S", "%m/%d/%Y %I:%M:%S %p"):
            try:
                return datetime.strptime(ph, fmt)
            except ValueError:
                continue
    return datetime.now()


def poll_esp32_device():
    global esp32_status, power_history, voltage_history, current_history
    prune_counter = 0
    while True:
        try:
            resp = requests.get(f"{ESP32_DEVICE_BASE_URL}info", timeout=3)
            if resp.status_code == 200:
                parsed = parse_esp32_response(resp.text)

                # Use the ESP32's own NTP clock for the timestamp
                dt = _esp32_dt(parsed)

                with esp32_lock:
                    esp32_status.update(parsed)
                    esp32_status["error"]       = None
                    esp32_status["last_update"] = dt.strftime("%H:%M:%S")

                    # Maintain rolling history for charts
                    power_history.append(parsed.get("power", 0))
                    voltage_history.append(parsed.get("voltage", 0))
                    current_history.append(parsed.get("current", 0))
                    if len(power_history)   > MAX_HISTORY: power_history.pop(0)
                    if len(voltage_history) > MAX_HISTORY: voltage_history.pop(0)
                    if len(current_history) > MAX_HISTORY: current_history.pop(0)

                # 1. Save to SQLite
                db_log_energy("705",
                              parsed.get("voltage",    0),
                              parsed.get("current",    0),
                              parsed.get("power",      0),
                              parsed.get("energy_kwh", 0),
                              parsed.get("light_state","OFF"))


                prune_counter += 1
                if prune_counter % 10 == 0:
                    db_log_device_status(parsed.get("wifi",    "Unknown"),
                                         parsed.get("cloud",   "Unknown"),
                                         parsed.get("sd_card", "Unknown"))
                if prune_counter >= 3600:
                    prune_counter = 0
                    Thread(target=db_prune_old_data, daemon=True).start()

        except requests.exceptions.RequestException as e:
            with esp32_lock:
                esp32_status["error"]       = str(e)
                esp32_status["last_update"] = time.strftime("%H:%M:%S")
        time.sleep(1)


# ============================================================================
# LIGHTS AUTOMATION
# ============================================================================

# How long (seconds) after the last person detection before lights turn off
PERSON_GONE_TIMEOUT = 300  # 5 minutes

# Shared state for the auto-light controller
_last_person_time  = 0.0   # epoch time of last person detection
_auto_light_lock   = Lock()
auto_light_status  = {
    "mode":      "SCHEDULE",   # SCHEDULE | PERSON | OFF
    "event":     "Starting…",
    "countdown": 0,
}


def _is_during_scheduled_event():
    """Return (True, event_name) if right now falls inside a scheduled class, else (False, None)."""
    try:
        now          = datetime.now()
        current_time = now.strftime('%H:%M')
        day_name     = now.strftime('%A')
        schedule     = load_schedule_by_day("705", day_name)
        for event in schedule:
            start = event.get('start_time', '')
            end   = event.get('end_time',   '')
            if start and end and start <= current_time < end:
                return True, event.get('event_name', 'Class')
    except Exception as e:
        print(f"[LIGHTS] Schedule check error: {e}")
    return False, None


def _turn_lights_on(reason: str):
    try:
        requests.get(f"{ESP32_DEVICE_BASE_URL}person", timeout=1)
        print(f"[LIGHTS] 💡 ON  — {reason}")
    except Exception as e:
        print(f"[LIGHTS] Error turning ON: {e}")


def _turn_lights_off(reason: str):
    try:
        requests.get(f"{ESP32_DEVICE_BASE_URL}toggle", timeout=1)
        print(f"[LIGHTS] 🔴 OFF — {reason}")
    except Exception as e:
        print(f"[LIGHTS] Error turning OFF: {e}")


def auto_light_controller():
    """
    Runs every 0.5s. Priority order:
      1. SCHEDULE — if a class is active, lights ON regardless of camera.
      2. PERSON   — if a person was detected recently (within PERSON_GONE_TIMEOUT), lights ON.
      3. OFF      — no schedule + no recent person → lights OFF.
    """
    global _last_person_time

    _prev_light_on    = False   # tracks last commanded state to avoid spamming ESP32
    _prev_person_seen = False   # for edge-detection (person appeared / person left)

    print("[AUTO-LIGHT] Controller started")

    while True:
        time.sleep(0.5)

        now        = time.time()
        person_now = person_count_global > 0

        # ── Edge detection: log person appeared / left ────────────────────
        if person_now and not _prev_person_seen:
            print("[AUTO-LIGHT] 👤 Person detected")
        elif not person_now and _prev_person_seen:
            print(f"[AUTO-LIGHT] 🚶 Person left — waiting {PERSON_GONE_TIMEOUT}s before OFF")
        _prev_person_seen = person_now

        # ── Update last-seen timestamp ────────────────────────────────────
        if person_now:
            with _auto_light_lock:
                _last_person_time = now

        # ── Decide what to do ─────────────────────────────────────────────
        during_class, event_name = _is_during_scheduled_event()
        with _auto_light_lock:
            last_seen = _last_person_time

        person_recently = (now - last_seen) < PERSON_GONE_TIMEOUT if last_seen > 0 else False
        gone_for        = now - last_seen if last_seen > 0 else 0
        countdown       = max(0, PERSON_GONE_TIMEOUT - gone_for) if not person_now and last_seen > 0 else 0

        if during_class:
            # ── Priority 1: Always ON during a scheduled class ────────────
            with _auto_light_lock:
                auto_light_status.update({
                    "mode":      "SCHEDULE",
                    "event":     f"Class: {event_name}",
                    "countdown": 0,
                })
            if not _prev_light_on:
                _turn_lights_on(f"scheduled class '{event_name}'")
                _prev_light_on = True

        elif person_recently or person_now:
            # ── Priority 2: Person detected (or just left, within timeout) ─
            with _auto_light_lock:
                auto_light_status.update({
                    "mode":      "PERSON",
                    "event":     "Person detected" if person_now else f"Person left — OFF in {int(countdown)}s",
                    "countdown": int(countdown),
                })
            if not _prev_light_on:
                _turn_lights_on("person present")
                _prev_light_on = True

        else:
            # ── Priority 3: No class + no person → OFF ────────────────────
            with _auto_light_lock:
                auto_light_status.update({
                    "mode":      "OFF",
                    "event":     "No class, no person",
                    "countdown": 0,
                })
            if _prev_light_on:
                _turn_lights_off("no class and no person detected")
                _prev_light_on = False


def lights_automation_loop():
    """Legacy 60-second schedule checker — kept as a safety net fallback."""
    print("[✓] Schedule fallback loop started")
    while True:
        try:
            time.sleep(300)   # every 5 minutes as a safety net
            during_class, event_name = _is_during_scheduled_event()
            if during_class:
                print(f"[LIGHTS-FALLBACK] Ensuring ON for '{event_name}'")
                _turn_lights_on(f"fallback: {event_name}")
        except Exception as e:
            print(f"[LIGHTS FALLBACK] Error: {e}")
            time.sleep(60)


# ============================================================================
# CAMERA LOOP
# ============================================================================

def camera_loop():
    global shared_frame, person_count_global, avg_confidence_global, camera_connected, last_notification_time
    print(f"[CAMERA] Connecting to: {ESP32_CAM_STREAM_URL}")
    while True:
        cap = None
        try:
            cap = cv2.VideoCapture(ESP32_CAM_STREAM_URL)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                print("[CAMERA] Failed to open stream, retrying in 5s...")
                camera_connected = False
                time.sleep(5)
                continue
            print("[CAMERA] Connected to ESP32-CAM stream")
            camera_connected = True
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[CAMERA] Frame read failed, reconnecting...")
                    camera_connected = False
                    break
                small   = cv2.resize(frame, (640, 480))
                persons = 0
                confidences = []
                if yolo_model is not None and detection_enabled:
                    results = yolo_model(small, verbose=False, device=device)
                    h, w    = frame.shape[:2]
                    for r in results:
                        for box in r.boxes:
                            cls  = int(box.cls[0])
                            conf = float(box.conf[0])
                            if cls != 0:
                                continue   # not a person class
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            x1 = int(x1 * w / 640); y1 = int(y1 * h / 480)
                            x2 = int(x2 * w / 640); y2 = int(y2 * h / 480)
                            if conf < PERSON_CONFIDENCE_THRESHOLD:
                                # Draw grey box — detected but below threshold, ignored
                                cv2.rectangle(frame, (x1,y1),(x2,y2),(128,128,128),2)
                                cv2.putText(frame, f"? {int(conf*100)}% (ignored)",
                                            (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128,128,128), 1)
                                continue   # does NOT count as a person
                            # Passes threshold — count it
                            persons += 1
                            confidences.append(conf)
                            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,215,255),3)
                            cv2.putText(frame, f"Person {int(conf*100)}%",
                                        (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,215,255), 2)
                person_count_global    = persons
                avg_confidence_global  = int(sum(confidences)/len(confidences)*100) if confidences else 0
                status_text = f"Persons: {persons} | {'OCCUPIED' if persons>0 else 'VACANT'} | Min conf: {int(PERSON_CONFIDENCE_THRESHOLD*100)}%"
                color = (0,255,0) if persons == 0 else (0,0,255)
                cv2.putText(frame, status_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                if persons > 0:
                    now = time.time()
                    if now - last_notification_time > DETECTION_COOLDOWN:
                        last_notification_time = now
                        try:
                            requests.get(f"{ESP32_DEVICE_BASE_URL}person", timeout=1)
                        except Exception:
                            pass
                with frame_lock:
                    shared_frame = frame.copy()
                time.sleep(0.05)
        except Exception as e:
            print(f"[CAMERA] Error: {e}")
            camera_connected = False
        finally:
            if cap is not None:
                cap.release()
            time.sleep(5)


# ============================================================================
# FLASK ROUTES — PAGES
# ============================================================================

@app.route('/')
def dashboard():
    return render_template('dashboard.html',
                           available_rooms=['705'],
                           default_date=date.today().isoformat(),
                           default_room='705')


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


# ============================================================================
# CAMERA ROUTES
# ============================================================================

@app.route('/video_feed')
def video_feed():
    if CLOUD_MODE or cv2 is None:
        return jsonify({'error': 'Camera disabled in cloud mode'}), 503
    def generate():
        while True:
            with frame_lock:
                frame = shared_frame.copy() if shared_frame is not None else None
            if frame is None:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Connecting to ESP32-CAM...", (80,240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
                _, buffer = cv2.imencode('.jpg', placeholder)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.2)
                continue
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.05)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/camera/stats')
def camera_stats():
    return jsonify({
        'person_count':               person_count_global,
        'avg_confidence':             avg_confidence_global,
        'occupied':                   person_count_global > 0,
        'camera_connected':           camera_connected,
        'confidence_threshold':       int(PERSON_CONFIDENCE_THRESHOLD * 100),
        'timestamp':                  datetime.now().isoformat()
    })


@app.route('/api/camera/confidence-threshold', methods=['GET', 'POST'])
def set_confidence_threshold():
    global PERSON_CONFIDENCE_THRESHOLD
    if request.method == 'POST':
        val = request.json.get('threshold', 70)
        PERSON_CONFIDENCE_THRESHOLD = max(0.1, min(0.99, float(val) / 100))
        print(f"[CAMERA] Confidence threshold set to {int(PERSON_CONFIDENCE_THRESHOLD*100)}%")
    return jsonify({'threshold': int(PERSON_CONFIDENCE_THRESHOLD * 100)})


@app.route('/api/camera/status')
def camera_status():
    return jsonify({'status': 'connected' if camera_connected else 'disconnected',
                    'timestamp': datetime.now().isoformat()})


@app.route('/api/camera/detection')
def toggle_detection():
    global detection_enabled
    detection_enabled = request.args.get('enabled', 'true').lower() == 'true'
    return jsonify({'detection_enabled': detection_enabled})


# ============================================================================
# ENERGY ROUTES (live SQLite)
# ============================================================================

@app.route('/api/energy/latest')
def get_latest_energy():
    room    = request.args.get('room', '705')
    reading = db_get_latest_reading(room)
    if reading:
        return jsonify({
            'success': True, 'room': room,
            'voltage':    round(reading.get('voltage',    0), 2),
            'current':    round(reading.get('current',    0), 6),
            'power':      round(reading.get('power',      0), 2),
            'energy_kwh': round(reading.get('energy_kwh', 0), 3),
            'light_state': reading.get('light_state', 'OFF'),
            'timestamp':   reading.get('timestamp', ''),
        })
    return jsonify({'success': False, 'error': 'No data available', 'room': room}), 404


@app.route('/api/energy/hourly')
def get_hourly_energy():
    room  = request.args.get('room', '705')
    hours = int(request.args.get('hours', 12))
    data  = db_get_hourly_data(room, hours)
    if data:
        labels     = [row['hour'].split(' ')[1] for row in data]
        power_data = [round(row['avg_power'], 2) for row in data]
        return jsonify({'success': True, 'room': room,
                        'labels': labels, 'power_data': power_data, 'raw_data': data})
    return jsonify({'success': False, 'error': 'No data available', 'room': room}), 404


@app.route('/api/esp32/status')
def get_esp32_status():
    with esp32_lock:
        status = dict(esp32_status)
    return jsonify(status)


@app.route('/api/energy/export')
def export_energy_csv():
    room     = request.args.get('room', '705')
    days     = int(request.args.get('days', 7))
    filename = db_export_to_csv(room, days)
    if filename:
        return jsonify({'success': True, 'filename': filename,
                        'message': f'Exported {days} days for Room {room}'})
    return jsonify({'success': False, 'error': 'No data to export'}), 404



# ============================================================================
# ENERGY CHART + PREDICTIONS (DB only)
# ============================================================================

@app.route('/api/energy/chart')
@app.route('/api/energy/live-chart')
def energy_chart():
    """
    15-minute bucketed power chart from SQLite for any date.
    Works for today (live) and historical dates equally.
    """
    date_param = request.args.get('date', datetime.now().strftime('%Y-%m-%d')).strip()
    room       = request.args.get('room', '705')

    day_start = f"{date_param} 00:00:00"
    day_end   = f"{date_param} 23:59:59"

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT timestamp, voltage, current, power, energy_kwh
               FROM energy_log
               WHERE room=? AND timestamp >= ? AND timestamp <= ?
               ORDER BY timestamp ASC""",
            (room, day_start, day_end)
        ).fetchall()
        conn.close()
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

    if not rows:
        # No data for this date — tell the frontend which dates are available
        try:
            conn = sqlite3.connect(DB_PATH)
            date_rows = conn.execute(
                "SELECT DISTINCT DATE(timestamp) as d FROM energy_log WHERE room=? ORDER BY d ASC",
                (room,)
            ).fetchall()
            conn.close()
            available = [r[0] for r in date_rows]
        except Exception:
            available = []
        return jsonify({
            'success':         False,
            'error':           f'No data for {date_param}',
            'available_dates': available,
            'source':          'db'
        }), 404

    return energy_chart_for_date(date_param, room, rows)


def energy_chart_for_date(date_str, room, rows=None):
    """Build chart JSON for a given date. Fetches rows from DB if not provided."""
    if rows is None:
        day_start = f"{date_str} 00:00:00"
        day_end   = f"{date_str} 23:59:59"
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT timestamp, voltage, current, power, energy_kwh
                   FROM energy_log
                   WHERE room=? AND timestamp >= ? AND timestamp <= ?
                   ORDER BY timestamp ASC""",
                (room, day_start, day_end)
            ).fetchall()
            conn.close()
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    buckets = defaultdict(list)
    for r in rows:
        try:
            dt    = datetime.strptime(r['timestamp'], '%Y-%m-%d %H:%M:%S')
            m     = dt.hour * 60 + dt.minute
            start = (m // 15) * 15
            label = f"{start // 60:02d}:{start % 60:02d}"
            buckets[label].append(r['power'])
        except Exception:
            continue

    sorted_b   = sorted(buckets.items())
    labels     = [b[0] for b in sorted_b]
    power_data = [round(sum(b[1]) / len(b[1]), 2) for b in sorted_b]

    all_power   = [r['power']      for r in rows]
    all_voltage = [r['voltage']    for r in rows]
    all_current = [r['current']    for r in rows]
    all_energy  = [r['energy_kwh'] for r in rows]

    return jsonify({
        'success':    True,
        'date':       date_str,
        'source':     'db',
        'labels':     labels,
        'power_data': power_data,
        'summary': {
            'avg_power':    round(sum(all_power)   / len(all_power),   2),
            'max_power':    round(max(all_power),   2),
            'avg_voltage':  round(sum(all_voltage) / len(all_voltage), 2),
            'avg_current':  round(sum(all_current) / len(all_current), 6),
            'total_energy': round(max(all_energy)  - min(all_energy),  4),
            'data_points':  len(rows),
        }
    })


@app.route('/api/energy/available-dates')
def available_dates():
    """Return all dates that have data in the DB, for the date picker."""
    room = request.args.get('room', '705')
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT DISTINCT DATE(timestamp) as d FROM energy_log WHERE room=? ORDER BY d ASC",
            (room,)
        ).fetchall()
        conn.close()
        return jsonify({'success': True, 'dates': [r[0] for r in rows]})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/predict-energy')
def predict_energy():
    """
    Predict hourly energy for a target date using historical DB data.
    Groups by day-of-week. If that weekday historically shows ~0W, returns zeros.
    """
    date_param = request.args.get('date', '').strip()
    room_param = request.args.get('room', '705')

    if not date_param:
        return jsonify({'error': 'date parameter required'}), 400

    try:
        target_dt = datetime.strptime(date_param, '%Y-%m-%d')
    except ValueError:
        return jsonify({'error': 'Invalid date format, use YYYY-MM-DD'}), 400

    target_weekday = target_dt.weekday()
    dow_labels     = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

    # Pull all historical rows from DB grouped by weekday
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        # All rows for this room
        all_rows = conn.execute(
            "SELECT timestamp, power, energy_kwh FROM energy_log WHERE room=? ORDER BY timestamp ASC",
            (room_param,)
        ).fetchall()
        conn.close()
    except Exception as e:
        return jsonify({'error': f'DB error: {e}'}), 500

    if not all_rows:
        return jsonify({'error': 'No historical data in database yet'}), 404

    # Parse timestamps and group by weekday
    parsed_rows = []
    for r in all_rows:
        try:
            dt = datetime.strptime(r['timestamp'], '%Y-%m-%d %H:%M:%S')
            parsed_rows.append({'dt': dt, 'power': r['power'], 'energy_kwh': r['energy_kwh']})
        except Exception:
            continue

    same_dow_rows = [r for r in parsed_rows if r['dt'].weekday() == target_weekday]

    # Zero-day check: if this weekday averages below 5W historically → predict 0
    ZERO_THRESHOLD_W = 5.0
    dow_avg_power = (sum(r['power'] for r in same_dow_rows) / len(same_dow_rows)) if same_dow_rows else 0.0
    is_zero_day   = dow_avg_power < ZERO_THRESHOLD_W

    print(f"[PREDICT] {dow_labels[target_weekday]} | DB rows: {len(same_dow_rows)} | avg: {dow_avg_power:.2f}W | zero: {is_zero_day}")

    if is_zero_day:
        return jsonify({
            'success':              True,
            'date':                 date_param,
            'day_of_week':          dow_labels[target_weekday],
            'room':                 room_param,
            'room_type':            'Laboratory' if room_param.startswith('7') else 'Office/Classroom',
            'is_zero_day':          True,
            'total_daily_kWh':      0.0,
            'predicted_rate':       11.0,
            'total_cost_Php':       0.0,
            'avg_hourly_kWh':       0.0,
            'avg_hourly_cost_Php':  0.0,
            'hourly_breakdown':     [0.0] * 24,
            'hourly_power_w':       [0.0] * 24,
            'confidence':           95,
            'recommendations':      [
                f"{dow_labels[target_weekday]}s show no energy consumption in historical data.",
                "No classes or activities are scheduled on this day of the week.",
                "Predicted consumption: 0 kWh."
            ],
            'data_source':          'energy_dashboard.db',
            'historical_rows_used': len(parsed_rows),
        })

    # Build hourly averages — weight same-weekday rows 2x
    hourly_power = defaultdict(list)
    for r in parsed_rows:
        weight = 2 if r['dt'].weekday() == target_weekday else 1
        for _ in range(weight):
            hourly_power[r['dt'].hour].append(r['power'])

    hourly_avg_w = []
    for h in range(24):
        vals = hourly_power.get(h, [])
        avg  = (sum(vals) / len(vals)) if vals else dow_avg_power
        hourly_avg_w.append(round(avg, 2))

    hourly_kwh = [round(w / 1000, 4) for w in hourly_avg_w]
    total_kwh  = round(sum(hourly_kwh), 4)
    rate       = 11.0
    total_cost = round(total_kwh * rate, 2)
    confidence = min(95, 60 + len(same_dow_rows) // 10)

    recommendations = []
    peak_hour  = hourly_avg_w.index(max(hourly_avg_w))
    peak_power = max(hourly_avg_w)
    if peak_power > 500:
        recommendations.append(f"Peak usage expected around {peak_hour:02d}:00 ({peak_power:.0f} W).")
    if total_kwh > 5:
        recommendations.append(f"Projected {total_kwh} kWh — consider turning off AC during unscheduled hours.")
    if total_kwh <= 1:
        recommendations.append("Low energy day projected based on historical patterns.")
    if not recommendations:
        recommendations.append(f"Normal {dow_labels[target_weekday]} usage projected.")

    return jsonify({
        'success':              True,
        'date':                 date_param,
        'day_of_week':          dow_labels[target_weekday],
        'room':                 room_param,
        'room_type':            'Laboratory' if room_param.startswith('7') else 'Office/Classroom',
        'is_zero_day':          False,
        'total_daily_kWh':      total_kwh,
        'predicted_rate':       rate,
        'total_cost_Php':       total_cost,
        'avg_hourly_kWh':       round(total_kwh / 24, 4),
        'avg_hourly_cost_Php':  round(total_cost / 24, 2),
        'hourly_breakdown':     hourly_kwh,
        'hourly_power_w':       hourly_avg_w,
        'confidence':           confidence,
        'recommendations':      recommendations,
        'data_source':          'energy_dashboard.db',
        'historical_rows_used': len(parsed_rows),
    })



@app.route('/api/predictive-rooms')
def predictive_rooms():
    """Return rooms that actually have energy data in the DB."""
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT DISTINCT room FROM energy_log ORDER BY room ASC"
        ).fetchall()
        conn.close()
        rooms = [r[0] for r in rows] if rows else ['705']
        return jsonify({'rooms': rooms})
    except Exception as e:
        return jsonify({'rooms': ['705'], 'error': str(e)})


@app.route('/api/auto-light/status')
def auto_light_status_api():
    with _auto_light_lock:
        status = dict(auto_light_status)
    status['person_count']       = person_count_global
    status['person_gone_timeout'] = PERSON_GONE_TIMEOUT
    with _auto_light_lock:
        last = _last_person_time
    status['last_person_seen'] = datetime.fromtimestamp(last).strftime('%H:%M:%S') if last > 0 else 'Never'
    return jsonify(status)


# ============================================================================
# SCHEDULE ROUTES
# ============================================================================

@app.route('/api/schedule/<room>/<day>')
def get_schedule_by_day(room, day):
    try:
        schedule_data = load_schedule_by_day(room, day)
        return jsonify(schedule_data if schedule_data else [])
    except Exception as e:
        print(f"[!] Schedule API error: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.route('/api/health')
def health_check():
    with esp32_lock:
        esp_ok = esp32_status.get('error') is None
    return jsonify({
        'status': 'healthy', 'service': 'Energy Dashboard',
        'camera_connected': camera_connected, 'esp32_ok': esp_ok,
        'database': 'OK', 'timestamp': datetime.now().isoformat(), 'version': '3.5.0'
    })




# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("ENERGY DASHBOARD v4.0 — DB-only mode")
    print("=" * 70)

    init_db()
    import_csv_to_db()   # one-time migration of energy_data.csv → DB (safe to re-run)

    print(f"Dashboard URL    : http://localhost:5000")
    print(f"ESP32-CAM Stream : {ESP32_CAM_STREAM_URL}")
    print(f"ESP32 Device     : {ESP32_DEVICE_BASE_URL}")
    print(f"Chart API        : http://localhost:5000/api/energy/chart")
    print(f"Predict API      : http://localhost:5000/api/predict-energy?date=2026-03-02&room=705")
    print(f"Available Dates  : http://localhost:5000/api/energy/available-dates")
    print(f"Schedule API     : http://localhost:5000/api/schedule/705/Monday")
    print(f"Database         : {DB_PATH}")
    print("=" * 70)

    cam_thread = Thread(target=camera_loop, daemon=True)
    cam_thread.start()
    print("[✓] Camera thread started")

    esp32_thread = Thread(target=poll_esp32_device, daemon=True)
    esp32_thread.start()
    print("[✓] ESP32 polling thread started")

    auto_light_thread = Thread(target=auto_light_controller, daemon=True)
    auto_light_thread.start()
    print("[✓] Auto-light controller started")

    lights_thread = Thread(target=lights_automation_loop, daemon=True)
    lights_thread.start()
    print("[✓] Schedule fallback loop started")

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)