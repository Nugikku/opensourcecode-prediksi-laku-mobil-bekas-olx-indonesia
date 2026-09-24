import os
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error

TANGGAL_HARI_INI = datetime.now().strftime("%d-%m-%Y")
OUTPUT_DIR = f"output_regresi_{TANGGAL_HARI_INI}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")

print("Membaca dataset bersih...")
file_dataset = "dataset_olx_bersih.csv"
if not os.path.exists(file_dataset): 
    raise FileNotFoundError(f"File '{file_dataset}' tidak ditemukan. Jalankan processing.py terlebih dahulu.")

df = pd.read_csv(file_dataset)

# Fitur untuk estimasi nilai pasar wajar (Fair Market Value)
categorical_cols = ['merek', 'model', 'transmisi', 'tipe_penjual', 'ada_garansi']
numerical_cols = ['tahun', 'jarak_tempuh', 'usia_mobil', 'km_per_tahun']
fitur = categorical_cols + numerical_cols
target = 'harga'

df_model = df.dropna(subset=fitur + [target]).copy()
X = df_model[fitur]
y = df_model[target]

print(f"Total sampel latih : {len(df_model):,} baris")
print(f"Fitur Kategorik     : {categorical_cols}")
print(f"Fitur Numerik       : {numerical_cols}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', StandardScaler(), numerical_cols)
])

# 5 Algoritma Regresi Komparasi
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0, random_state=42),
    "Decision Tree": DecisionTreeRegressor(max_depth=12, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42)
}

hasil_metrik = []
trained_pipelines = {}
best_model_name, best_model_pipeline, best_r2 = None, None, -np.inf

print("\n" + "=" * 75)
print("PELATIHAN & EVALUASI 5 ALGORITMA REGRESI HARGA PASAR")
print("=" * 75)

for nama_model, regressor in models.items():
    pipeline = Pipeline(steps=[('prep', preprocessor), ('reg', regressor)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    trained_pipelines[nama_model] = (pipeline, y_pred)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    print(f"\nModel: {nama_model}")
    print(f"  R2-Score : {r2:.4f} ({r2 * 100:.2f}% variansi harga terjelaskan)")
    print(f"  MAE      : Rp {mae:,.0f}")
    print(f"  RMSE     : Rp {rmse:,.0f}")
    print(f"  MAPE     : {mape:.2f}%")

    hasil_metrik.append({
        "Model": nama_model,
        "R2_Score": r2,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    })

    if r2 > best_r2:
        best_r2, best_model_name, best_model_pipeline = r2, nama_model, pipeline

print("\n" + "=" * 75)
print(f"Model Regresi Terbaik Terpilih: {best_model_name} (R2-Score: {best_r2:.4f})")
print("=" * 75)

df_metrik = pd.DataFrame(hasil_metrik)

# Simpan Model Terbaik
with open("model_regresi_harga.pkl", "wb") as f:
    pickle.dump(best_model_pipeline, f)
print(f"[SUKSES] Model regresi tersimpan ke 'model_regresi_harga.pkl'")

# ============================================================
# PEMBUATAN 4 GRAFIK EVALUASI REGRESI DETAIL
# ============================================================
print(f"\nMenghasilkan grafik visualisasi ke folder '{OUTPUT_DIR}/'...")

# --- 1. Komparasi R2-Score & MAPE Semua Model ---
fig, ax1 = plt.subplots(figsize=(10, 5))
x = np.arange(len(df_metrik))
lebar = 0.35

ax1.bar(x - lebar/2, df_metrik["R2_Score"], lebar, label="R2-Score", color="#2563EB")
ax1.set_ylabel("R2-Score", color="#2563EB", fontsize=11, fontweight="bold")
ax1.set_ylim(0, 1.1)

ax2 = ax1.twinx()
ax2.bar(x + lebar/2, df_metrik["MAPE"], lebar, label="MAPE (%)", color="#DC2626")
ax2.set_ylabel("Error Relatif / MAPE (%)", color="#DC2626", fontsize=11, fontweight="bold")
ax2.set_ylim(0, max(df_metrik["MAPE"]) * 1.3)
ax2.grid(False)

ax1.set_xticks(x)
ax1.set_xticklabels(df_metrik["Model"], rotation=15, ha="right", fontsize=10, fontweight="bold")
ax1.set_title("Komparasi Performa Algoritma Regresi (R2 vs MAPE)", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "1_komparasi_metrik_regresi.png"), dpi=300)
plt.close(fig)

# --- 2. Scatter Plot Aktual vs Prediksi (Model Terbaik) ---
_, y_pred_best = trained_pipelines[best_model_name]
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test / 1e6, y_pred_best / 1e6, alpha=0.45, color="#059669", edgecolors="none")

# Garis Ideal (Identity line)
min_val = min(y_test.min(), y_pred_best.min()) / 1e6
max_val = max(y_test.max(), y_pred_best.max()) / 1e6
ax.plot([min_val, max_val], [min_val, max_val], color="#DC2626", linestyle="--", lw=2, label="Prediksi Sempurna (Y = X)")

ax.set_xlabel("Harga Aktual Pasar (Juta Rupiah)", fontsize=11)
ax.set_ylabel("Harga Prediksi Model (Juta Rupiah)", fontsize=11)
ax.set_title(f"Akurasi Prediksi Harga — {best_model_name} (R2: {best_r2:.3f})", fontsize=13, pad=10)
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "2_actual_vs_predicted.png"), dpi=300)
plt.close(fig)

# --- 3. Distribusi Error Residual ---
residual = (y_test - y_pred_best) / 1e6
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(residual, bins=40, color="#6366F1", edgecolor="black", alpha=0.7)
ax.axvline(0, color="#DC2626", linestyle="--", lw=2, label="Titik Nol Error")
ax.set_xlabel("Selisih Error: Aktual - Prediksi (Juta Rupiah)", fontsize=11)
ax.set_ylabel("Frekuensi Unit Mobil", fontsize=11)
ax.set_title(f"Distribusi Residual Error — {best_model_name}", fontsize=13, pad=10)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "3_distribusi_residual.png"), dpi=300)
plt.close(fig)

# --- 4. Feature Importance Top 15 ---
reg_final = best_model_pipeline.named_steps['reg']
if hasattr(reg_final, "feature_importances_"):
    importances = reg_final.feature_importances_
    nama_fitur = best_model_pipeline.named_steps['prep'].get_feature_names_out()
    df_imp = pd.DataFrame({"Fitur": nama_fitur, "Importance": importances}).sort_values("Importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df_imp["Fitur"][::-1], df_imp["Importance"][::-1], color="#0D9488")
    ax.set_xlabel("Skor Kepentingan Relatif (Feature Importance)", fontsize=11)
    ax.set_title(f"15 Fitur Paling Berpengaruh dalam Penentuan Harga — {best_model_name}", fontsize=13, pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "4_feature_importance_regresi.png"), dpi=300)
    plt.close(fig)

# Simpan Ringkasan CSV
df_metrik.to_csv(os.path.join(OUTPUT_DIR, "ringkasan_metrik_regresi.csv"), index=False)
print(f"[SELESAI] Model dan seluruh visualisasi regresi berhasil disimpan ke folder '{OUTPUT_DIR}/'.")