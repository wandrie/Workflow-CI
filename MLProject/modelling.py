import os
import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================
# MLFLOW CONFIGURATION
# ============================================

if os.getenv("GITHUB_ACTIONS"):
    # Di GitHub, simpan log ke folder lokal (default mlruns)
    mlflow.set_tracking_uri(None) 
else:
# Set tracking URI ke localhost
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Set experiment name
mlflow.set_experiment("Sistem Pendeteksi Harga Rumah di California")

# ============================================
# LOAD DATA
# ============================================

def load_processed_data():
    base_dir = Path(__file__).parent.resolve()
    train_path = base_dir / "train_processed.csv"
    test_path = base_dir / "test_processed.csv"
        
    if not train_path.exists():
        raise FileNotFoundError(
            f"❌ Error: File tidak ditemukan. Pastikan data preprocessing diletakkan di folder MLProject."
        )
        
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # Pisahkan Fitur (X) dan Target (y)
    X_train = df_train.drop(columns=['MedHouseVal'])
    y_train = df_train['MedHouseVal']
    
    X_test = df_test.drop(columns=['MedHouseVal'])
    y_test = df_test['MedHouseVal']
    
    print(f"✅ [Autolog] Data Latih Dimuat: {X_train.shape}")
    print(f"✅ [Autolog] Data Uji Dimuat  : {X_test.shape}")
    return X_train, X_test, y_train, y_test

# ============================================
# LOG ARTIFACTS
# ============================================

def log_custom_plots(y_test, y_pred, model, feature_names):
    """Fungsi kustom untuk tetap mencatat visualisasi evaluasi tambahan ke MLflow Artifacts"""
    # Residual Plot
    plt.figure(figsize=(10, 6))
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5, color='purple')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Prediksi Harga')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.savefig('residual_plot.png')
    plt.close()
    mlflow.log_artifact('residual_plot.png')

    # Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='teal')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Harga Aktual')
    plt.ylabel('Prediksi Harga')
    plt.title('Harga Aktual vs Prediksi Harga')
    plt.savefig('actual_vs_prediksi.png')
    plt.close()
    mlflow.log_artifact('actual_vs_prediksi.png')

    # Feature importance (hanya untuk model yang mendukung)
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importance')
        plt.bar(range(len(importance)), importance[indices], color='royalblue')
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        mlflow.log_artifact('feature_importance.png')

# ============================================
# TRAINING LOGIC (PURE AUTOLOG VERSION)
# ============================================

def train_with_autolog(X_train, X_test, y_train, y_test, model, model_name):
    mlflow.sklearn.autolog(log_models=True)
    
    def execute_training():
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        log_custom_plots(y_test, y_pred_test, model, X_train.columns)
        r2_test = model.score(X_test, y_test)
        return model, r2_test

    # DETEKSI LINGKUNGAN CI/CD
    is_ci_cd = os.getenv("GITHUB_ACTIONS") is not None

    if is_ci_cd or mlflow.active_run():
        print(f" -> CI/CD terdeteksi atau Run aktif ditemukan: {mlflow.active_run().info.run_id}")
        return execute_training()
    
    else:
        with mlflow.start_run(run_name=model_name):
            return execute_training()
    
    # ============================================
    # MAIN
    # ============================================
if __name__ == "__main__":
    print("="*50)
    print("STARTING EXPERIMENT: PURE MLFLOW AUTOLOG PIPELINE")
    print("="*50)
    
    # Muat data hasil pembersihan awal
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Kandidat algoritma model
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Random Forest': RandomForestRegressor(),
        'Gradient Boosting': GradientBoostingRegressor()
    }
    
    best_model = None
    best_r2 = -np.inf
    best_model_name = ""
    
    # Menjalankan iterasi pelatihan otomatis
    for model_name, model_obj in models.items():
        print(f"\n⏳ [Autolog] Melatih & Merekam otomatis: {model_name}...")
        trained_model, r2_score_test = train_with_autolog(X_train, X_test, y_train, y_test, model_obj, model_name)
        
        # Menentukan model terbaik untuk diekspor ke lokal produksi
        if r2_score_test > best_r2:
            best_r2 = r2_score_test
            best_model = trained_model
            best_model_name = model_name
            
    # Simpan salinan model untuk kebutuhan production serving/inference API
    output_dir = "saved_models"
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(output_dir, 'best_model.pkl'))
    
    print("\n" + "="*30)
    print(f"🏆 Pemenang Eksperimen: {best_model_name}")
    print(f"📈 R² Test Score      : {best_r2*100:.2f}%")
    print(f"💾 Model sukses dicatat di MLflow & diekspor ke '{output_dir}/best_model.pkl'")
    print("="*50)