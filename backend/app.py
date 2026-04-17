from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import psutil
import time
import threading
import numpy as np
from sklearn.ensemble import IsolationForest
import os
import json

app = FastAPI()

# ====== CORS ======
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== DATA ======
metrics_data = []
MAX_DATA = 100

# ====== MODEL ======
model = IsolationForest(contamination=0.1)

# ====== LOGGING ======
def log_event(event):
    with open("logs.txt", "a") as f:
        f.write(json.dumps(event) + "\n")

# ====== MONITORING ======
def collect_metrics():
    while True:
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent

        metrics_data.append([cpu, memory])

        if len(metrics_data) > MAX_DATA:
            metrics_data.pop(0)

        time.sleep(2)

# ====== TRAIN MODEL ======
def train_model():
    while True:
        if len(metrics_data) >= 10:
            data = np.array(metrics_data)
            model.fit(data)
        time.sleep(10)

threading.Thread(target=collect_metrics, daemon=True).start()
threading.Thread(target=train_model, daemon=True).start()

# ====== ROUTES ======

@app.get("/")
def home():
    return {"message": "Running"}

@app.get("/metrics")
def get_metrics():
    if len(metrics_data) == 0:
        return {"data": []}

    data = np.array(metrics_data)

    if len(metrics_data) >= 10:
        preds = model.predict(data)
    else:
        preds = [1] * len(metrics_data)

    result = []
    for i in range(len(metrics_data)):
        result.append({
            "cpu": metrics_data[i][0],
            "memory": metrics_data[i][1],
            "anomaly": preds[i] == -1
        })

    return {"data": result}

@app.get("/anomaly")
def detect_anomaly():
    if len(metrics_data) < 10:
        return {"anomaly": False}

    data = np.array(metrics_data)
    preds = model.predict(data)

    is_anomaly = preds[-1] == -1

    if is_anomaly:
        log_event({
            "event": "ANOMALY",
            "data": metrics_data[-1],
            "time": time.time()
        })

    return {"anomaly": is_anomaly}

# 🔥 FIXED PREDICTION LOGIC
@app.get("/predict")
def predict_failure():
    if len(metrics_data) < 6:
        return {"prediction": "Collecting data..."}

    last_cpu = [m[0] for m in metrics_data[-5:]]

    # simple trend: compare average
    avg_start = sum(last_cpu[:2]) / 2
    avg_end = sum(last_cpu[-2:]) / 2

    if avg_end > avg_start + 10:  # threshold
        log_event({
            "event": "PREDICTION",
            "trend": last_cpu,
            "time": time.time()
        })

        return {"prediction": "⚠️ CPU trend increasing → Risk of failure"}

    return {"prediction": "✅ System stable"}

@app.get("/self-heal")
def self_heal():
    cpu = psutil.cpu_percent()

    if cpu > 80:
        os.system("echo restart simulated")

        log_event({
            "event": "SELF_HEAL",
            "cpu": cpu,
            "time": time.time()
        })

        return {"action": "⚠️ Restart triggered (simulated)"}

    return {"action": "✅ Stable"}