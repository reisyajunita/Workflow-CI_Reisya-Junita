import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn # Penting untuk autolog sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # Contoh model sederhana
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os

# --- 1. Fungsi Pemuatan Data ---
def load_processed_data(processed_data_path):
    """Memuat data yang sudah sepenuhnya diproses dari Kriteria 1."""
    try:
        df = pd.read_csv(processed_data_path)
        print(f"Data berhasil dimuat dari: {processed_data_path}, Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File '{processed_data_path}' tidak ditemukan.")
        return None

# --- 2. Fungsi Utama untuk Training dan Logging (Basic dengan Autolog) ---
def train_model_basic_autolog(df_processed, model_choice, target_col='Churn'):
    """
    Melakukan train-test split, melatih model dasar,
    dan menggunakan MLflow autolog untuk mencatat eksperimen.
    """
    if df_processed is None:
        print("Proses training dibatalkan karena data tidak tersedia.")
        return

    if target_col not in df_processed.columns:
        print(f"Error: Kolom target '{target_col}' tidak ditemukan.")
        return

    X = df_processed.drop(target_col, axis=1)
    y = df_processed[target_col] # Diasumsikan target sudah 0/1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Data di-split: X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # Aktifkan MLflow autologging untuk scikit-learn
    # Ini akan otomatis mencatat parameter, metrik, dan model
    mlflow.sklearn.autolog(
        log_model_signatures=True,  # Mencatat signature input/output model
        log_input_examples=True,    # Mencatat contoh input data
        registered_model_name=f"{model_choice.__class__.__name__}ChurnModelBasic"
    )

    # Set nama eksperimen MLflow (akan dibuat jika belum ada di folder mlruns)
    experiment_name = f"TelcoChurn_{model_choice.__class__.__name__}_Basic_Autolog"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"Run_{model_choice.__class__.__name__}_Basic") as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        print(f"Experiment Name: {mlflow.get_experiment(run.info.experiment_id).name}")
        print(f"MLflow Artifact URI (lokal): {mlflow.get_artifact_uri()}")

        print(f"\nMelatih model: {model_choice.__class__.__name__}...")
        # Latih model
        model_choice.fit(X_train, y_train)
        print("Model berhasil dilatih.")

        # Evaluasi model pada test set (autolog akan mencatat metrik ini)
        y_pred_test = model_choice.predict(X_test)
        
        # Hitung metrik secara manual untuk ditampilkan di console (opsional, karena autolog sudah mencatatnya)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        precision_test = precision_score(y_test, y_pred_test, zero_division=0)
        recall_test = recall_score(y_test, y_pred_test, zero_division=0)
        f1_test = f1_score(y_test, y_pred_test, zero_division=0)
        
        # predict_proba untuk roc_auc
        try:
            y_pred_proba_positive_class_test = model_choice.predict_proba(X_test)[:, 1]
            roc_auc_test = roc_auc_score(y_test, y_pred_proba_positive_class_test)
            print(f"ROC AUC (Test): {roc_auc_test:.4f}")
        except AttributeError:
            print("Model tidak memiliki predict_proba, ROC AUC mungkin tidak dicatat oleh autolog dari probabilitas.")
            roc_auc_test = "N/A"


        print(f"\nMetrics on Test Set (juga dicatat oleh autolog):")
        print(f"Accuracy: {accuracy_test:.4f}")
        print(f"Precision: {precision_test:.4f}")
        print(f"Recall: {recall_test:.4f}")
        print(f"F1 Score: {f1_test:.4f}")
        
        print(f"\nMLflow Autologging aktif. Parameter, metrik, dan model akan otomatis tercatat.")
        print(f"MLflow Run Selesai. Run ID: {run_id}")
        print(f"Lihat hasil di MLflow UI (jalankan 'mlflow ui' di terminal pada direktori '{os.getcwd()}' atau direktori yang berisi 'mlruns').")

# --- 3. Main Execution Block ---
if __name__ == "__main__":
    # Path ke data yang sudah diproses oleh Kriteria 1
    input_data_path = "telco-dataset_preprocessing/dataset_processed.csv"

    df_processed = load_processed_data(input_data_path)

    if df_processed is not None:
        # --- Contoh Eksperimen dengan Logistic Regression ---
        print("\n--- Starting Experiment: Logistic Regression (Basic Autolog) ---")
        # Inisialisasi model dengan parameter default atau parameter sederhana
        logreg_basic_model = LogisticRegression(solver='liblinear', random_state=42, max_iter=200) 
        
        train_model_basic_autolog(df_processed.copy(), logreg_basic_model)
    else:
        print("Pemrosesan model dibatalkan karena data tidak dimuat.")