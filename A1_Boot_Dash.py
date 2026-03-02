# ============================================================================
# A1_Boot_Dash.py — COMPLETE FILE (v3.5 — energy_data.csv integration)
# ============================================================================

from flask import Flask, render_template, send_from_directory, jsonify, request, Response
import os
import cv2
import torch
import time
import numpy as np
import pandas as pd
import sqlite3
import csv
import io
import re
from datetime import datetime, date, timedelta
from threading import Thread, Lock
from collections import defaultdict
import sys
import requests

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

ESP32_CAM_IP         = "192.168.254.103"
ESP32_CAM_STREAM_URL = f"http://{ESP32_CAM_IP}/stream"
ESP32_DEVICE_IP      = "192.168.254.118"
ESP32_DEVICE_BASE_URL = f"http://{ESP32_DEVICE_IP}/"
DETECTION_COOLDOWN   = 2

# Lock so the CSV file is never written by two threads at once
_csv_write_lock  = Lock()

# ============================================================================
# YOLO MODEL
# ============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[✓] Using device: {device}")

yolo_model = None
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


def db_prune_old_data(keep_days=30):
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

# ============================================================================
# ENERGY_DATA.CSV — parsed ONCE at startup, cached in memory
# ============================================================================

ENERGY_CSV_PATH = os.path.join(current_dir, 'assets', 'energy_data.csv')

# Global cache — populated by _load_csv_cache() at startup
_CSV_ROWS        = []          # list of dicts: {dt, date_str, voltage, current, power, energy_kwh}
_CSV_DATES       = []          # sorted list of 'YYYY-MM-DD' strings
_CSV_DATES_SET   = set()
_CSV_BY_DATE     = {}          # 'YYYY-MM-DD' -> list of row dicts
_CSV_LOADED      = False


def _load_csv_cache():
    """Parse energy_data.csv once using pandas and build lookup structures."""
    global _CSV_ROWS, _CSV_DATES, _CSV_DATES_SET, _CSV_BY_DATE, _CSV_LOADED

    print(f"[CSV] Loading: {ENERGY_CSV_PATH}")
    if not os.path.exists(ENERGY_CSV_PATH):
        print(f"[CSV] ❌ File not found: {ENERGY_CSV_PATH}")
        return

    try:
        df = pd.read_csv(ENERGY_CSV_PATH, encoding='utf-8-sig', dtype=str)

        # Strip BOM and whitespace from column names
        df.columns = [c.strip().lstrip('\ufeff') for c in df.columns]
        print(f"[CSV] Columns : {list(df.columns)}")
        print(f"[CSV] Rows    : {len(df)}")
        print(f"[CSV] Sample  : {df.iloc[0].to_dict()}")

        # ── Extract year directly from the raw Date string (e.g. "1/21/2026" → 2026)
        # This filters bad rows BEFORE pandas datetime ever touches them
        df['_raw_year'] = pd.to_numeric(
            df['Date'].str.strip().str.split('/').str[-1].str.strip(),
            errors='coerce'
        )
        bad_year_pre = ~df['_raw_year'].between(2000, 2099)
        if bad_year_pre.any():
            print(f"[CSV] ⚠ Dropped {bad_year_pre.sum()} rows with bad raw year: {df.loc[bad_year_pre, '_raw_year'].dropna().unique().tolist()}")
        df = df[~bad_year_pre].copy()

        # Build datetime string then parse with EXPLICIT format
        # Data looks like: Date="1/21/2026"  Time="1:39:52 PM"
        dt_str = df['Date'].str.strip() + ' ' + df['Time'].str.strip()

        # Try the format that matches the actual data first
        df['_dt'] = pd.to_datetime(dt_str, format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

        # Fallback: try 24-hour format for any rows that failed
        mask_failed = df['_dt'].isna()
        if mask_failed.any():
            df.loc[mask_failed, '_dt'] = pd.to_datetime(
                dt_str[mask_failed], format='%m/%d/%Y %H:%M:%S', errors='coerce'
            )

        bad = df['_dt'].isna().sum()
        if bad:
            print(f"[CSV] ⚠ Dropped {bad} unparseable rows")
        df = df.dropna(subset=['_dt'])

        # Final sanity check on parsed year
        bad_year = ~df['_dt'].dt.year.between(2000, 2099)
        if bad_year.any():
            print(f"[CSV] ⚠ Dropped {bad_year.sum()} rows with bad parsed years: {df.loc[bad_year, '_dt'].dt.year.unique().tolist()}")
            df = df[~bad_year]

        # Numeric columns
        for col in ['Voltage(V)', 'Current(A)', 'Power(W)', 'Energy(kWh)']:
            df[col] = pd.to_numeric(df.get(col, 0), errors='coerce').fillna(0)

        df['_date_str'] = df['_dt'].dt.strftime('%Y-%m-%d')

        print(f"[CSV] Dates found: {sorted(df['_date_str'].unique().tolist())}")

        # ── Build cache using vectorized ops (no iterrows) ──
        _CSV_DATES     = sorted(df['_date_str'].unique().tolist())
        _CSV_DATES_SET = set(_CSV_DATES)

        # Convert to list of dicts via records — ~100x faster than iterrows
        records = df[['_dt', '_date_str', 'Voltage(V)', 'Current(A)', 'Power(W)', 'Energy(kWh)']].to_dict('records')

        _CSV_ROWS = [
            {
                'dt':         r['_dt'],
                'date_str':   r['_date_str'],
                'voltage':    float(r['Voltage(V)']),
                'current':    float(r['Current(A)']),
                'power':      float(r['Power(W)']),
                'energy_kwh': float(r['Energy(kWh)']),
            }
            for r in records
        ]

        # Group by date for O(1) per-request lookup
        for r in _CSV_ROWS:
            d = r['date_str']
            if d not in _CSV_BY_DATE:
                _CSV_BY_DATE[d] = []
            _CSV_BY_DATE[d].append(r)

        _CSV_LOADED = True
        print(f"[CSV] ✅ Cached {len(_CSV_ROWS)} rows across {len(_CSV_DATES)} dates")

    except Exception as e:
        print(f"[CSV] ❌ Error: {e}")
        import traceback; traceback.print_exc()


def get_csv_rows(date_str=None):
    """Return cached rows, optionally filtered to a single date_str ('YYYY-MM-DD')."""
    if date_str:
        return _CSV_BY_DATE.get(date_str, [])
    return _CSV_ROWS


# ============================================================================
# SCHEDULE LOADING
# ============================================================================

def load_schedule_by_day(room, day_name):
    """Load schedule from templates/room_schedules.csv"""
    try:
        schedule_file = os.path.join(current_dir, 'templates', 'room_schedules.csv')
        print(f"[SCHEDULE] Looking for CSV at: {schedule_file}")
        print(f"[SCHEDULE] File exists: {os.path.exists(schedule_file)}")

        if not os.path.exists(schedule_file):
            print(f"[SCHEDULE] ❌ File NOT found. current_dir={current_dir}")
            return []

        schedule_data = []
        with open(schedule_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            print(f"[SCHEDULE] CSV columns: {reader.fieldnames}")
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

        print(f"[SCHEDULE] ✅ Found {len(schedule_data)} events for Room {room} on {day_name}")
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

                # 2. Append to energy_data.csv using ESP32's NTP timestamp
                _append_to_energy_csv({
                    'dt':         dt,
                    'voltage':    parsed.get("voltage",    0),
                    'current':    parsed.get("current",    0),
                    'power':      parsed.get("power",      0),
                    'energy_kwh': parsed.get("energy_kwh", 0),
                })

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
                            cls = int(box.cls[0])
                            if cls == 0:
                                persons += 1
                                conf     = float(box.conf[0])
                                confidences.append(conf)
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                x1 = int(x1 * w / 640); y1 = int(y1 * h / 480)
                                x2 = int(x2 * w / 640); y2 = int(y2 * h / 480)
                                cv2.rectangle(frame, (x1,y1),(x2,y2),(0,215,255),3)
                                cv2.putText(frame, f"Person {int(conf*100)}%", (x1,y1-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,215,255), 2)
                person_count_global    = persons
                avg_confidence_global  = int(sum(confidences)/len(confidences)*100) if confidences else 0
                status_text = f"Persons: {persons} | {'OCCUPIED' if persons>0 else 'VACANT'}"
                color = (0,255,0) if persons == 0 else (0,0,255)
                cv2.putText(frame, status_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
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
        'person_count': person_count_global, 'avg_confidence': avg_confidence_global,
        'occupied': person_count_global > 0, 'camera_connected': camera_connected,
        'timestamp': datetime.now().isoformat()
    })


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
# ENERGY_DATA.CSV ROUTES (historical chart + predictions)
# ============================================================================

@app.route('/api/debug/csv')
def debug_csv():
    return jsonify({
        'path':            ENERGY_CSV_PATH,
        'exists':          os.path.exists(ENERGY_CSV_PATH),
        'loaded':          _CSV_LOADED,
        'rows_cached':     len(_CSV_ROWS),
        'available_dates': _CSV_DATES,
        'sample':          {
            'dt':      _CSV_ROWS[0]['dt'].strftime('%Y-%m-%d %H:%M:%S'),
            'power':   _CSV_ROWS[0]['power'],
            'voltage': _CSV_ROWS[0]['voltage'],
        } if _CSV_ROWS else None
    })


@app.route('/api/energy/live-chart')
def live_chart():
    """
    Real-time chart data from SQLite for a specific date.
    Buckets readings into 15-minute intervals — same format as csv-chart.
    Used by the Overview tab for today's date so new readings appear instantly.
    """
    date_param = request.args.get('date', datetime.now().strftime('%Y-%m-%d')).strip()
    room       = request.args.get('room', '705')

    # Date window: full day from 00:00:00 to 23:59:59
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
        return jsonify({
            'success': False,
            'error':   f'No DB data for {date_param}',
            'source':  'db'
        }), 404

    # 15-minute bucket aggregation — same logic as csv-chart
    from collections import defaultdict
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
        'date':       date_param,
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


@app.route('/api/energy/csv-chart')
def csv_chart():
    """Returns 15-min bucketed power data for the chart from the in-memory cache."""
    if not _CSV_LOADED:
        return jsonify({'success': False, 'error': 'CSV cache not loaded'}), 503

    date_param = request.args.get('date', '').strip()
    if not date_param:
        # Pick the date in the CSV closest to today without going over
        today_str = datetime.now().strftime('%Y-%m-%d')
        past_dates = [d for d in _CSV_DATES if d <= today_str]
        target_str = past_dates[-1] if past_dates else (_CSV_DATES[0] if _CSV_DATES else '')
    else:
        target_str = date_param if date_param in _CSV_DATES_SET else ''
        if not target_str:
            today_str = datetime.now().strftime('%Y-%m-%d')
            past_dates = [d for d in _CSV_DATES if d <= today_str]
            target_str = past_dates[-1] if past_dates else (_CSV_DATES[0] if _CSV_DATES else '')

    if not target_str:
        return jsonify({'success': False, 'error': 'No data available'}), 404

    day_rows = _CSV_BY_DATE.get(target_str, [])
    if not day_rows:
        return jsonify({'success': False, 'error': f'No data for {target_str}', 'available_dates': _CSV_DATES}), 404

    # 15-minute buckets
    buckets = defaultdict(list)
    for r in day_rows:
        m     = r['dt'].hour * 60 + r['dt'].minute
        start = (m // 15) * 15
        label = f"{start // 60:02d}:{start % 60:02d}"
        buckets[label].append(r['power'])

    sorted_b   = sorted(buckets.items())
    labels     = [b[0] for b in sorted_b]
    power_data = [round(sum(b[1]) / len(b[1]), 2) for b in sorted_b]

    all_power   = [r['power']      for r in day_rows]
    all_voltage = [r['voltage']    for r in day_rows]
    all_current = [r['current']    for r in day_rows]
    all_energy  = [r['energy_kwh'] for r in day_rows]

    return jsonify({
        'success':         True,
        'date':            target_str,
        'available_dates': _CSV_DATES,
        'labels':          labels,
        'power_data':      power_data,
        'summary': {
            'avg_power':    round(sum(all_power)   / len(all_power),   2),
            'max_power':    round(max(all_power),   2),
            'avg_voltage':  round(sum(all_voltage) / len(all_voltage), 2),
            'avg_current':  round(sum(all_current) / len(all_current), 6),
            'total_energy': round(max(all_energy)  - min(all_energy),  4),
            'data_points':  len(day_rows),
        }
    })


@app.route('/api/predict-energy')
def predict_energy():
    """Predict hourly energy using historical CSV data.
    If the target day-of-week has negligible consumption in history, predicts 0.
    """
    date_param = request.args.get('date', '').strip()
    room_param = request.args.get('room', '705')

    if not date_param:
        return jsonify({'error': 'date parameter required'}), 400
    if not _CSV_LOADED:
        return jsonify({'error': 'CSV cache not loaded yet'}), 503

    try:
        target_dt = datetime.strptime(date_param, '%Y-%m-%d')
    except ValueError:
        return jsonify({'error': 'Invalid date format, use YYYY-MM-DD'}), 400

    target_weekday = target_dt.weekday()   # 0=Mon … 6=Sun
    dow_labels     = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

    # ── Step 1: Gather all rows that share the same day-of-week ──────────────
    same_dow_rows = [r for r in _CSV_ROWS if r['dt'].weekday() == target_weekday]

    # ── Step 2: Check if that day-of-week has meaningful consumption ─────────
    # A day is considered "zero consumption" if the average power across ALL
    # historical readings for that weekday is below a noise threshold (5 W).
    ZERO_THRESHOLD_W = 5.0

    if same_dow_rows:
        dow_avg_power = sum(r['power'] for r in same_dow_rows) / len(same_dow_rows)
    else:
        dow_avg_power = 0.0

    is_zero_day = dow_avg_power < ZERO_THRESHOLD_W

    print(f"[PREDICT] {dow_labels[target_weekday]} | historical rows: {len(same_dow_rows)} | avg power: {dow_avg_power:.2f} W | zero_day: {is_zero_day}")

    # ── Step 3: If zero day, return all zeros immediately ────────────────────
    if is_zero_day:
        recommendations = [
            f"{dow_labels[target_weekday]}s show no energy consumption in historical data — room is unoccupied.",
            "No classes or activities are scheduled on this day of the week.",
            "Predicted consumption: 0 kWh."
        ]
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
            'recommendations':      recommendations,
            'data_source':          'energy_data.csv',
            'historical_rows_used': len(_CSV_ROWS),
        })

    # ── Step 4: Active day — build hourly averages from same-weekday rows ────
    # Weight same-weekday rows 2× vs all other rows for accuracy
    hourly_power = defaultdict(list)
    for r in _CSV_ROWS:
        weight = 2 if r['dt'].weekday() == target_weekday else 1
        for _ in range(weight):
            hourly_power[r['dt'].hour].append(r['power'])

    # For any hour with no data at all, use the same-weekday average
    dow_global_avg = dow_avg_power

    hourly_avg_w = []
    for h in range(24):
        vals = hourly_power.get(h, [])
        avg  = (sum(vals) / len(vals)) if vals else dow_global_avg
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
        recommendations.append(f"Peak usage expected around {peak_hour:02d}:00 ({peak_power:.0f} W) — consider load scheduling.")
    if total_kwh > 5:
        recommendations.append(f"Projected {total_kwh} kWh — consider turning off AC during unscheduled hours.")
    if total_kwh <= 1:
        recommendations.append("Low energy day projected based on historical patterns.")
    if not recommendations:
        recommendations.append(f"Normal {dow_labels[target_weekday]} usage projected based on historical patterns.")

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
        'data_source':          'energy_data.csv',
        'historical_rows_used': len(_CSV_ROWS),
    })


@app.route('/api/predictive-rooms')
def predictive_rooms():
    return jsonify({'rooms': ['101', '102', '705']})


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
    print("ENERGY DASHBOARD v3.5 — STARTING")
    print("=" * 70)

    init_db()
    _load_csv_cache()   # parse energy_data.csv once into memory

    print(f"Dashboard URL    : http://localhost:5000")
    print(f"ESP32-CAM Stream : {ESP32_CAM_STREAM_URL}")
    print(f"ESP32 Device     : {ESP32_DEVICE_BASE_URL}")
    print(f"CSV Debug        : http://localhost:5000/api/debug/csv")
    print(f"CSV Chart API    : http://localhost:5000/api/energy/csv-chart")
    print(f"Predict API      : http://localhost:5000/api/predict-energy?date=2026-01-21&room=705")
    print(f"Schedule API     : http://localhost:5000/api/schedule/705/Monday")
    print(f"Database         : {DB_PATH}")
    print(f"Energy CSV       : {ENERGY_CSV_PATH}")
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