# ⚡ Lumergy — Smart Energy Dashboard

A real-time IoT energy monitoring and management dashboard for university classrooms, built with Flask. Lumergy integrates ESP32 hardware sensors, YOLOv8 person detection, and machine-learning forecasting to provide intelligent lighting automation and energy consumption insights.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1-lightgrey?logo=flask)
![YOLOv8](https://img.shields.io/badge/YOLO-v8n-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Features

| Tab | Description |
|-----|-------------|
| **Overview** | Live voltage, current, power, and energy (kWh) readings from ESP32 sensors. Sparkline charts, hourly aggregates, and CSV exports. |
| **Predictive** | ML-powered energy consumption forecasts (XGBoost) and electricity rate predictions (Holt-Winters) up to 2027. |
| **Camera** | Real-time ESP32-CAM stream with YOLOv8 person detection; auto-light control when people enter/leave the room. |
| **Schedule** | Room schedule timeline loaded from CSV; lights turn on/off automatically per scheduled classes. |

### Highlights

- **Automatic Lighting** — Three-tier priority system: Schedule -> Person Detection -> OFF.
- **Night Mode** — Toggle dark theme for the entire dashboard.
- **Energy Forecasting** — Predicts daily kWh and estimated costs using trained ML models.
- **Rate Forecasting** — Holt-Winters time-series model projects electricity rates up to 2027.
- **CSV Import/Export** — Bulk-import historical data on startup; export filtered readings anytime.
- **SQLite Database** — Local, zero-config persistence with automatic pruning of old data.

---

## Project Structure

```
├── A1_Boot_Dash.py           # Main Flask application (routes, threads, ESP32 logic)
├── wsgi.py                   # Production WSGI entry point (Gunicorn / Render)
├── PA1_data_preprocessing.py # Data cleaning & feature engineering
├── PA2_model_training.py     # Model training pipeline
├── PA3_energy_predictor.py   # Energy consumption prediction engine
├── PA4_rate_forecaster.py    # Electricity rate forecasting (Holt-Winters)
├── initialize_forecaster.py  # One-time forecaster initialization
├── requirements.txt          # Python dependencies
├── render.yaml               # Render deployment blueprint
├── render-build.sh           # Render build script
├── yolov8n.pt                # YOLOv8 nano model weights
├── *.joblib                  # Pre-trained ML models
├── assets/
│   ├── energy_data.csv       # Historical energy readings (imported to DB on first run)
│   ├── energy_rate.csv       # Historical electricity rates
│   ├── room_schedules.csv    # Room schedule definitions
│   ├── sample_data.csv       # Sample training data
│   ├── notifications.csv     # System notifications
│   ├── recommendations.csv   # Energy-saving recommendations
│   └── sched_database.csv    # Schedule database
├── exports/                  # Generated CSV exports
├── static/
│   ├── D1_bootstrap.min.css  # Bootstrap 5
│   ├── D2_dashboard.css      # Custom dashboard styles
│   ├── D3_bootstrap.bundle.min.js
│   └── custom.css
└── templates/
    ├── dashboard.html        # Main layout (sidebar + tab container)
    ├── overview_content.html # Overview tab
    ├── predictive_content.html
    ├── camera_content.html
    └── schedule_content.html
```

---

## Prerequisites

- **Python 3.10+** (3.11 recommended)
- **ESP32-CAM** module for camera streaming (optional for dashboard-only mode)
- **ESP32 energy sensor** module with voltage/current/power readings (optional)
- Both ESP32 devices must be on the **same local network** as the server

---

## Local Development

### 1. Clone the repository

```bash
git clone <copy_link>
cd lumergy-dashboard
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate    # Linux / macOS
# venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure ESP32 IPs (optional)

Edit the top of `A1_Boot_Dash.py` if your ESP32 devices have different IPs

### 5. Run the dashboard

```bash
python A1_Boot_Dash.py
```

Open **http://localhost:5000** in your browser.

> On first startup the app imports `assets/energy_data.csv` into SQLite (this may take a few seconds for large datasets). Subsequent startups skip already-imported rows.

---

## Deploying to Render

[Render](https://render.com) is the recommended hosting platform — it supports long-running Python web services, background threads, and persistent disk storage (all required by this app).

### Quick Deploy (Blueprint)

1. Push the repository to GitHub.
2. Go to [dashboard.render.com](https://dashboard.render.com) -> **New** -> **Blueprint**.
3. Connect your GitHub repo.
4. Render will auto-detect `render.yaml` and configure everything.
5. Click **Apply** — your dashboard will be live in a few minutes.

### Manual Deploy

1. Go to [dashboard.render.com](https://dashboard.render.com) -> **New** -> **Web Service**.
2. Connect your GitHub repo.
3. Configure:

| Setting | Value |
|---------|-------|
| **Runtime** | Python |
| **Build Command** | `./render-build.sh` |
| **Start Command** | `gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120` |
| **Plan** | Free (or Starter for better performance) |

4. Under **Environment**, add:
   - `PYTHON_VERSION` = `3.11.9`
5. *(Optional)* Attach a **Disk** (1 GB, mounted at `/data`) for persistent SQLite storage — **requires a paid plan (Starter+)**. On the free tier, the DB is rebuilt from CSV on each deploy.
6. Click **Deploy**.

### Important Notes for Cloud Deployment

- **ESP32 hardware features** (live sensor data, camera stream, automatic lighting) require the ESP32 devices to be on the same network as the server. These features will **not work** when deployed to the cloud — the dashboard will still load and display historical/predicted data.
- **Free tier** services spin down after 15 minutes of inactivity. The first request after a spin-down takes ~30 seconds.
- **Free tier** does not support persistent disks — the SQLite database is rebuilt from `assets/energy_data.csv` on every deploy. Upgrade to Starter+ and uncomment the `disk` section in `render.yaml` for persistence.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard page |
| `/video_feed` | GET | MJPEG camera stream |
| `/api/energy/latest` | GET | Latest sensor reading |
| `/api/energy/hourly` | GET | Hourly aggregated data |
| `/api/energy/chart` | GET | Chart data with date/room filters |
| `/api/energy/available-dates` | GET | Dates with recorded data |
| `/api/energy/export` | GET | Export data as CSV |
| `/api/esp32/status` | GET | ESP32 device status |
| `/api/camera/stats` | GET | Person count & confidence |
| `/api/camera/status` | GET | Camera connection status |
| `/api/camera/detection` | GET/POST | Toggle person detection |
| `/api/camera/confidence-threshold` | GET/POST | Get/set detection threshold |
| `/api/predict-energy` | GET | Energy prediction (`?date=&room=`) |
| `/api/predictive-rooms` | GET | Available rooms for prediction |
| `/api/auto-light/status` | GET | Auto-light controller state |
| `/api/schedule/<room>/<day>` | GET | Room schedule for a given day |
| `/api/health` | GET | Health check |

---

## Machine Learning Models

| Model File | Algorithm | Purpose |
|------------|-----------|---------|
| `kwh_predictor.joblib` | XGBoost | Predicts daily energy consumption (kWh) per room |
| `rate_forecaster.joblib` | Holt-Winters (ETS) | Forecasts electricity rates (₱/kWh) up to 2027 |
| `room_encoder.joblib` | Label Encoder | Encodes room identifiers for the ML pipeline |
| `room_type_encoder.joblib` | Label Encoder | Encodes room types (lab, lecture, etc.) |
| `model_registry.joblib` | Registry | Stores metadata about trained models |
| `yolov8n.pt` | YOLOv8 Nano | Real-time person detection from camera feed |

To retrain the energy models:

```bash
python PA2_model_training.py
```

---

## Tech Stack

- **Backend:** Flask 3.1, Gunicorn, SQLite
- **Frontend:** Bootstrap 5, Chart.js, Jinja2 templates
- **ML/AI:** XGBoost, scikit-learn, statsmodels (Holt-Winters), Ultralytics YOLOv8
- **Hardware:** ESP32-CAM (camera), ESP32 + PZEM-004T (energy sensor)
- **CV:** OpenCV (MJPEG stream processing + YOLO inference)
- **Deployment:** Render (recommended), Docker, or any Linux VPS

---

## Environment Variables (optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `ESP32_CAM_IP` | `172.20.10.5` | IP address of the ESP32-CAM module |
| `ESP32_DEVICE_IP` | `172.20.10.4` | IP address of the ESP32 energy sensor |
| `FLASK_ENV` | `production` | Flask environment mode |
| `PORT` | `5000` | Server port (set automatically by Render) |

---

## License

This project is for academic and educational purposes.
