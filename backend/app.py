from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import psutil
import time
import threading
from sklearn.ensemble import IsolationForest
import numpy as np

# ====== FASTAPI APP ======
app = FastAPI()

# ====== CORS (IMPORTANT for frontend) ======
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== DATA STORAGE ======
metrics_data = []

# ====== ML MODEL ======
model = IsolationForest(contamination=0.1)

# ====== MONITORING FUNCTION ======
def collect_metrics():
    while True:
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent

        data = [cpu, memory]
        metrics_data.append(data)

        # keep last 100 entries
        if len(metrics_data) > 100:
            metrics_data.pop(0)

        time.sleep(2)

# ====== TRAIN MODEL IN BACKGROUND ======
def train_model():
    while True:
        if len(metrics_data) >= 10:
            data = np.array(metrics_data)
            model.fit(data)
        time.sleep(10)

# Start threads
threading.Thread(target=collect_metrics, daemon=True).start()
threading.Thread(target=train_model, daemon=True).start()

# ====== ROUTES ======

@app.get("/")
def home():
    return {"message": "AI Self-Healing System Running"}

@app.get("/metrics")
def get_metrics():
    return {"data": metrics_data}

@app.get("/anomaly")
def detect_anomaly():
    if len(metrics_data) < 10:
        return {"status": "Not enough data"}

    data = np.array(metrics_data)
    preds = model.predict(data)

    return {"anomaly": preds[-1] == -1}

@app.get("/self-heal")
def self_heal():
    cpu = psutil.cpu_percent()

    if cpu > 80:
        return {
            "action": "⚠️ High CPU → Restart service (simulated)"
        }

    return {"action": "✅ System stable"}