import os
import time
import psutil
import requests
import pandas as pd
import numpy as np
from prometheus_client import start_http_server, Counter, Gauge, Histogram

# ============================================
# Menginisialisasi Metrik Prometheus
# ============================================
PREDICTION_COUNT = Counter(
    'model_predictions_total', 
    'Total number of predictions made by the model'
)

LAST_PREDICTION_VALUE = Gauge(
    'model_last_prediction_value', 
    'The last house price value predicted by the model'
)

PREDICTION_LATENCY = Histogram(
    'model_prediction_latency_seconds', 
    'Time taken to process prediction',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

CPU_USAGE = Gauge('system_cpu_usage', 'Persentase penggunaan CPU sistem secara real-time')
RAM_USAGE = Gauge('system_ram_usage', 'Persentase penggunaan RAM sistem secara real-time')

# URL endpoint serving MLflow yang sedang aktif di port 5002
URL_SERVING = "http://127.0.0.1:5002/invocations"

if __name__ == "__main__":
    print("==================================================")
    print("     PROMETHEUS EXPORTER FOR INFERENCE SERVICE    ")
    print("==================================================")
    
    # Menjalankan HTTP Server internal Prometheus di Port 8000
    PORT = 8001
    start_http_server(PORT)
    print(f"🚀 Exporter aktif! Menyiarkan metrik di http://localhost:{PORT}/metrics")
    print(f"🔗 Menghubungkan dan mengirim data simulasi ke MLflow Serving di Port 5002...")
    print("-" * 50)

    request_id = 1
    try:
        while True:
            # Membuat variasi nilai MedInc acak agar grafik pemantauan nanti fluktuatif
            random_medinc = np.random.uniform(2.0, 8.0)
            random_age = np.random.uniform(10.0, 52.0)
            
            # Format JSON payload standar untuk MLflow Serving
            payload = {
                "dataframe_split": {
                    "columns": ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude", "BedroomsPerRoom", "IncomeCategory_Encoded"],
                    "data": [[random_medinc, random_age, 5.4, 1.0, 1400.0, 2.8, 35.6, -119.5, 0.185, 1]]
                }
            }

            try:
                start_time = time.time()
                # Mengirim request POST ke MLflow Serving Port 5002
                response = requests.post(URL_SERVING, json=payload)
                duration = time.time() - start_time

                if response.status_code == 200:
                    hasil_prediksi = response.json()["predictions"][0]
                    
                    # Update Metrik Prometheus
                    PREDICTION_COUNT.inc()                  
                    LAST_PREDICTION_VALUE.set(hasil_prediksi) 
                    PREDICTION_LATENCY.observe(duration)     
                    
                    print(f"[{request_id}] Sukses Kirim ke 5002 -> Estimasi Harga Model: ${hasil_prediksi:.4f} | Latensi: {duration:.4f}s")
                else:
                    print(f"[{request_id}] Serving 5002 merespons dengan eror status: {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                print(f"[{request_id}] ❌ Gagal menyambung ke Port 5002. Pastikan server MLflow tetap menyala!")

            # Memperbarui metrik penggunaan CPU dan RAM secara real-time
            CPU_USAGE.set(psutil.cpu_percent())
            RAM_USAGE.set(psutil.virtual_memory().percent)
            
            request_id += 1
            time.sleep(3) # Kirim data simulasi setiap 3 detik sekali
            
    except KeyboardInterrupt:
        print("\n🛑 Exporter dihentikan secara manual.")
        print("==================================================")