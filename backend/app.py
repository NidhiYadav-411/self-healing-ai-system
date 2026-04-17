from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import psutil
import time
import threading
import numpy as np
from sklearn.ensemble import IsolationForest
import os
import json

# ====== INIT APP ======
app = FastAPI()

# ====== CORS ======
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== GLOBAL STORAGE ======
metrics_data = []
MAX_DATA = 100

# ====== ML MODEL ======
model = IsolationForest(contamination=0.1)

# ====== LOGGING FUNCTION ======
def log_event(event):
    with open("logs.txt", "a") as f:
        f.write(json.dumps(event) + "\n")

# ====== MONITORING FUNCTION ======
def collect_metrics():
    while True:
        try:
            cpu = psutil.cpu_percent()
            memory = psutil.virtual_memory().percent

            data = [cpu, memory]
            metrics_data.append(data)

            # keep only last 100 entries
            if len(metrics_data) > MAX_DATA:
                metrics_data.pop(0)

            time.sleep(2)

        except Exception as e:
            print("Monitoring error:", e)

# ====== MODEL TRAINING ======
def train_model():
    while True:
        try:
            if len(metrics_data) >= 10:
                data = np.array(metrics_data)
                model.fit(data)
            time.sleep(10)

        except Exception as e:
            print("Training error:", e)

# ====== START BACKGROUND THREADS ======
threading.Thread(target=collect_metrics, daemon=True).start()
threading.Thread(target=train_model, daemon=True).start()

# ====== ROUTES ======

@app.get("/")
def home():
    return {"message": "AI Self-Healing System Running"}

# ====== GET METRICS ======
@app.get("/metrics")
def get_metrics():
    return {"data": metrics_data}

# ====== ANOMALY DETECTION ======
@app.get("/anomaly")
def detect_anomaly():
    if len(metrics_data) < 10:
        return {"status": "Not enough data"}

    try:
        data = np.array(metrics_data)
        preds = model.predict(data)

        is_anomaly = preds[-1] == -1

        if is_anomaly:
            log_event({
                "event": "ANOMALY_DETECTED",
                "metrics": metrics_data[-1],
                "time": time.time()
            })

        return {"anomaly": is_anomaly}

    except Exception as e:
        return {"error": str(e)}

# ====== SELF-HEALING ======
@app.get("/self-heal")
def self_heal():
    try:
        cpu = psutil.cpu_percent()

        if cpu > 80:
            # ⚠️ SAFE SIMULATION (DO NOT actually kill processes yet)
            os.system("echo 'Simulated restart triggered'")

            log_event({
                "event": "SELF_HEAL_TRIGGERED",
                "cpu": cpu,
                "action": "restart_simulated",
                "time": time.time()
            })

            return {
                "action": "⚠️ High CPU → Restart command executed (simulated)"
            }

        return {"action": "✅ System stable"}

    except Exception as e:
        return {"error": str(e)}

# ====== VIEW LOGS ======
@app.get("/logs")
def get_logs():
    try:
        with open("logs.txt", "r") as f:
            logs = f.readlines()

        return {"logs": logs[-20:]}  # last 20 logs

    except:
        return {"logs": []}