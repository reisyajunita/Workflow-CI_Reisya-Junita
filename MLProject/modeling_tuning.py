import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub # Pastikan ini di-import
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, log_loss)
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Konfigurasi dan Inisialisasi DagsHub ---
# Untuk Git Bash / MINGW64
# export MLFLOW_TRACKING_URI="https://dagshub.com/reisyajunita/membangun_model.mlflow"
# export MLFLOW_TRACKING_USERNAME="reisyajunita"
# export MLFLOW_TRACKING_PASSWORD="TOKEN_"

# Environment Variables (MLFLOW_TRACKING_URI, USERNAME, PASSWORD/TOKEN)
# HARUS SUDAH DI-SET di terminal SEBELUM menjalankan script ini.

try:
    dagshub.init(repo_owner='reisyajunita', repo_name='membangun_model', mlflow=True)
    print("Panggilan dagshub.init() berhasil atau sudah terkonfigurasi.")
except ImportError:
    print("PERINGATAN: Library 'dagshub' belum terinstal. 'pip install dagshub'. Logging ke DagsHub mungkin gagal.")
except Exception as e:
    print(f"Peringatan saat dagshub.init(): {e}. Pastikan repo_owner & repo_name benar, dan env var MLFLOW sudah di-set.")

# Verifikasi Tracking URI
print(f"MLflow Tracking URI yang akan digunakan: {mlflow.get_tracking_uri()}")
if "dagshub.com" not in mlflow.get_tracking_uri():
    print("KRITIKAL: MLflow Tracking URI TIDAK mengarah ke DagsHub! Cek Environment Variables atau dagshub.init().")

# --- 1. Fungsi Pemuatan Data ---
def load_processed_data(processed_data_path):
    try:
        df = pd.read_csv(processed_data_path)
        print(f"Data berhasil dimuat dari: {processed_data_path}, Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File '{processed_data_path}' tidak ditemukan.")
        return None

# --- 2. Fungsi Utama untuk Training, Tuning, dan Logging Manual ke DagsHub ---
def train_model_with_tuning_dagshub(df_processed, model_choice, param_grid_model, input_path_for_logging, target_col='Churn', experiment_suffix="Advanced_DagsHub"):
    if df_processed is None:
        print("Proses training dibatalkan: data tidak tersedia.")
        return

    if target_col not in df_processed.columns:
        print(f"Error: Kolom target '{target_col}' tidak ditemukan.")
        return

    X = df_processed.drop(target_col, axis=1)
    y = df_processed[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Data di-split: X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    experiment_name = f"TelcoChurn_{model_choice.__class__.__name__}_{experiment_suffix}"
    try:
        current_experiment = mlflow.get_experiment_by_name(experiment_name)
        if current_experiment is None:
            # Saat membuat experiment baru untuk DagsHub, sertakan artifact_location
            mlflow.create_experiment(experiment_name, artifact_location=mlflow.get_artifact_uri())
        mlflow.set_experiment(experiment_name)
        print(f"Menggunakan eksperimen: '{experiment_name}' di DagsHub.")
    except Exception as e:
        print(f"Error saat mengatur eksperimen MLflow di DagsHub: {e}")
        return

    # Pastikan tidak ada run aktif sebelumnya sebelum memulai yang baru
    if mlflow.active_run():
        active_run_id = mlflow.active_run().info.run_id
        print(f"Mengakhiri run MLflow yang aktif sebelumnya: {active_run_id}")
        mlflow.end_run()

    with mlflow.start_run(run_name=f"Run_{model_choice.__class__.__name__}_{experiment_suffix.replace('_', '')}") as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        print(f"MLflow Artifact URI (DagsHub): {mlflow.get_artifact_uri()}")

        # Log parameter
        mlflow.log_param("input_data_path_for_run", input_path_for_logging) 
        mlflow.log_param("training_data_rows", X_train.shape[0])
        mlflow.log_param("training_data_cols", X_train.shape[1])
        mlflow.log_param("target_kriteria", "Advanced K2 DagsHub")

        print(f"\nMemulai GridSearchCV untuk {model_choice.__class__.__name__}...")
        grid_search = GridSearchCV(estimator=model_choice, param_grid=param_grid_model,
                                   cv=3, scoring='roc_auc', verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_cv_roc_auc = grid_search.best_score_
        print(f"Best Parameters: {best_params}")
        print(f"Best ROC AUC score from CV: {best_cv_roc_auc:.4f}")

        # MLflow Manual Logging
        mlflow.log_params(best_params)
        mlflow.log_param("model_class", model_choice.__class__.__name__)
        mlflow.log_param("cv_folds_gridsearch", grid_search.cv)
        mlflow.log_param("scoring_metric_cv_gridsearch", grid_search.scoring)

        y_pred_test = best_model.predict(X_test)
        y_pred_proba_test = best_model.predict_proba(X_test)
        y_pred_proba_positive_class_test = y_pred_proba_test[:, 1]

        accuracy_test = accuracy_score(y_test, y_pred_test)
        precision_test = precision_score(y_test, y_pred_test, zero_division=0)
        recall_test = recall_score(y_test, y_pred_test, zero_division=0)
        f1_test = f1_score(y_test, y_pred_test, zero_division=0)
        roc_auc_test = roc_auc_score(y_test, y_pred_proba_positive_class_test)
        logloss_test = log_loss(y_test, y_pred_proba_test) # Metrik Tambahan 1

        print(f"\nMetrics on Test Set: Acc: {accuracy_test:.4f}, Prec: {precision_test:.4f}, Rec: {recall_test:.4f}, F1: {f1_test:.4f}, ROC_AUC: {roc_auc_test:.4f}, LogLoss: {logloss_test:.4f}")
        mlflow.log_metric("accuracy_test", accuracy_test)
        mlflow.log_metric("precision_test", precision_test)
        mlflow.log_metric("recall_test", recall_test)
        mlflow.log_metric("f1_score_test", f1_test)
        mlflow.log_metric("roc_auc_test", roc_auc_test)
        mlflow.log_metric("log_loss_test", logloss_test)
        mlflow.log_metric("best_cv_roc_auc_train", best_cv_roc_auc)

        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train, y_pred_proba_test)
        input_example = X_train.head(3)

        print("\nLogging model ke DagsHub...")
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="tuned-churn-model-dagshub",
            signature=signature,
            input_example=input_example
        )
        print(f"Model '{model_choice.__class__.__name__}' logged ke DagsHub.")

        print("Logging Confusion Matrix ke DagsHub...")
        cm = confusion_matrix(y_test, y_pred_test)
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        ax.set_title('Confusion Matrix on Test Set'); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        mlflow.log_figure(fig, "evaluation_plots_dagshub/confusion_matrix_test.png")
        plt.close(fig)
        print(f"Confusion matrix plot logged ke DagsHub.")
        
        test_predictions_df = pd.DataFrame({'actual_churn': y_test, 'predicted_churn': y_pred_test, 'probability_churn': y_pred_proba_positive_class_test})
        mlflow.log_table(data=test_predictions_df, artifact_file="prediction_outputs_dagshub/test_set_predictions.json")
        print("Test set predictions logged sebagai table artifact ke DagsHub.")

        print(f"\nMLflow Run Selesai. Run ID: {run_id}. Cek DagsHub!")

# --- 3. Main Execution Block ---
if __name__ == "__main__":
    # Definisikan path data di sini agar bisa dilewatkan ke fungsi
    current_input_data_path = "telco-dataset_preprocessing/dataset_processed.csv" 
    
    df_processed = load_processed_data(current_input_data_path)

    if df_processed is not None:
        print("\n--- Starting Experiment (DagsHub): Logistic Regression ---")
        logreg_model = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
        param_grid_logreg = {
            'C': [0.1, 1, 10], 
            'penalty': ['l1', 'l2'], 
            'class_weight': [None, 'balanced']
        }
        train_model_with_tuning_dagshub(df_processed.copy(), logreg_model, param_grid_logreg, 
                                        input_path_for_logging=current_input_data_path, 
                                        experiment_suffix="Tuning_LR_Advanced")

        print("\n--- Starting Experiment (DagsHub): Random Forest ---")
        rf_model = RandomForestClassifier(random_state=42)
        param_grid_rf_simple = {
            'n_estimators': [100], 'max_depth': [10, 20],
            'min_samples_split': [5, 10], 'class_weight':['balanced', None]
        }
        train_model_with_tuning_dagshub(df_processed.copy(), rf_model, param_grid_rf_simple, 
                                        input_path_for_logging=current_input_data_path, 
                                        experiment_suffix="Tuning_RF_Advanced")
    else:
        print("Pemrosesan model dibatalkan.")
    
    print(f"\nSemua eksperimen selesai. Periksa hasil di DagsHub: https://dagshub.com/reisyajunita/membangun_model/experiments")