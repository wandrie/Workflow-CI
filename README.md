# 🏡 Sistem Pendeteksi Harga Rumah di California (End-to-End MLOps Pipeline)

Proyek ini merupakan implementasi pipa MLOps (_Machine Learning Operations_) lengkap dari hulu ke hilir, mulai dari tahap _preprocessing_ data, eksperimen otomatis & _hyperparameter tuning_ menggunakan MLflow, otomatisasi pengujian dengan GitHub Actions (CI), hingga tahap produksi berupa _serving_ model kustom dan _monitoring_ real-time berbasis Prometheus & Grafana.

---

## Pengembang

- **Nama:** Hervan Wandri
- **Peran:** MLOps / Data Engineer
- **Lingkungan Sistem:** macOS (MacBook Pro 2015) & Python 3.11

---

## 📁 Struktur Direktori Proyek

```text
SMSML_Wandrie/
├── README.md                           <- Panduan utama proyek (File ini)
├── Eksperimen_SML_Hervan-Wandri.txt    <- Tautan link github eksperimen
├── Inference.py                        <- Script pengujian prediksi data tunggal
├── Screenhoot_artifak.jpg              <- Bukti artifact mlflow ui
├── MLProject/
│   ├── california_housing_precessed    <- Berisi file csv yang sudah di proses
│   ├── modelling.py                    <- Base training (5 algoritma dasar)
│   ├── modelling_tuning.py             <- Hyperparameter Tuning (GridSearchCV)
│   ├── requirements.txt                <- Daftar dependensi pustaka Python
│   ├── test_processed.csv              <- Berisi file untuk data testing
│   └── train_processed.csv             <- Berisi file untuk data training
├── Monitoring dan Logging/
│    ├── 1.bukti_serving                 <- Log/bukti endpoint exporter aktif
│    ├── 2.prometheus.yml                <- Konfigurasi target scraping Prometheus
│    ├── 3.prometheus_exporter.py        <- Script kustom penyiaran metrik (psutil)
│    ├── 4.bukti monitoring Prometheus/  <- Screenshot metrik pada Prometheus UI
│    ├── 5.bukti monitoring Grafana/     <- Screenshot dashboard visualisasi RAM/CPU
│    └── 6.bukti alerting Grafana/       <- Screenshot pemicuan alarm (Alerting Firing)
└── Membangun_model/
   ├── california_housing_preprocessing/ <- Dataset hasil pembersihan & scaling
   ├── Eksperimen_Hervan-Wandri.ipynb    <- Berisi file eksperimen .ipynb
   ├── test_processed.csv                <- File test yang telah di ekspor
   └── train_processed.csv               <- File train_processed yang telah di ekspor
```
