"""
AutoLaku - Analisis & Seleksi Fitur (materi P3-4)

Alur:
  1. Baca dataset_olx_berlabel.csv, pakai DATA TRAINING saja.
  2. Korelasi & deteksi redundansi (mis. tahun vs usia_mobil ≈ korelasi sempurna).
  3. Metode filter: Mutual Information, Chi-Square.
  4. Metode wrapper: Recursive Feature Elimination (RFE) dengan kurva skor.
  5. Metode embedded: Feature Importance Random Forest, koefisien Lasso.
  6. Analisis TF-IDF pada deskripsi untuk kata pembeda label.
  7. Keluaran: keputusan_fitur.csv (bahan UTS nomor 3).
"""

import os
import sys
import json
import warnings

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import (mutual_info_classif, chi2,
                                       RFE, SelectKBest)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")
OUTPUT_DIR = "output_analisis_fitur"
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")

DATASET_BERLABEL = "dataset_olx_berlabel.csv"
KEPUTUSAN_FILE = "keputusan_fitur.csv"

LABEL_ORDER = ["Cepat", "Sedang", "Lambat"]
RANDOM_STATE = 42


def main():
    print("=" * 75)
    print("ANALISIS & SELEKSI FITUR (P3-4)")
    print("=" * 75)

    if not os.path.exists(DATASET_BERLABEL):
        raise FileNotFoundError(f"'{DATASET_BERLABEL}' tidak ditemukan. Jalankan pelabelan.py dulu.")

    df_all = pd.read_csv(DATASET_BERLABEL)
    df = df_all[df_all['split'] == 'train'].copy()
    print(f"Data training dimuat: {len(df):,} baris")

    # Fitur kandidat: semua fitur yang mungkin dipakai classifier
    categorical_cols = ['merek', 'model', 'transmisi', 'tipe_penjual', 'ada_garansi',
                        'riwayat_negatif', 'catatan_servis', 'pajak_hidup']
    numerical_cols = ['tahun', 'jarak_tempuh', 'usia_mobil', 'km_per_tahun']
    # Filter hanya kolom yang tersedia
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    numerical_cols = [c for c in numerical_cols if c in df.columns]
    all_features = categorical_cols + numerical_cols

    y = df['status_likuiditas']
    le_y = LabelEncoder()
    y_enc = le_y.fit_transform(y)

    keputusan = []  # akan menjadi keputusan_fitur.csv

    # ==============================================================
    # 1. KORELASI & DETEKSI REDUNDANSI
    # ==============================================================
    print("\n" + "=" * 75)
    print("1. KORELASI & DETEKSI REDUNDANSI")
    print("=" * 75)

    # Korelasi antar fitur numerikal
    num_data = df[numerical_cols].copy()
    if 'deviasi_persen' in df.columns:
        num_data['deviasi_persen'] = df['deviasi_persen']
    if 'indikasi_kredit_tdp' in df.columns:
        num_data['indikasi_kredit_tdp'] = df['indikasi_kredit_tdp']

    corr = num_data.corr()
    print("\nMatriks korelasi (numerikal):")
    print(corr.round(3).to_string())

    # Deteksi redundansi (|r| > 0.95)
    redundan = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            r = abs(corr.iloc[i, j])
            if r > 0.95:
                c1, c2 = corr.columns[i], corr.columns[j]
                redundan.append((c1, c2, r))
                print(f"  [REDUNDAN] {c1} vs {c2}: r = {r:.4f}")

    # Grafik heatmap korelasi
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr.columns, fontsize=9)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
    plt.colorbar(im)
    ax.set_title("Heatmap Korelasi Fitur Numerikal")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_heatmap_korelasi.png"), dpi=200)
    plt.close(fig)

    # Catat keputusan redundansi
    if redundan:
        for c1, c2, r in redundan:
            keputusan.append({
                "fitur": c2, "metode": "korelasi",
                "skor": f"r={r:.4f} dengan {c1}",
                "keputusan": "buang",
                "alasan": f"Redundan dengan {c1} (korelasi > 0.95). "
                          f"Contoh: tahun dan usia_mobil berkorelasi sempurna.",
            })
    else:
        print("  Tidak ditemukan pasangan fitur dengan korelasi > 0.95.")

    # ==============================================================
    # 2. METODE FILTER: MUTUAL INFORMATION
    # ==============================================================
    print("\n" + "=" * 75)
    print("2. METODE FILTER: MUTUAL INFORMATION")
    print("=" * 75)

    # Encode kategorik untuk MI
    prep_mi = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        ('num', StandardScaler(), numerical_cols),
    ])
    X_encoded = prep_mi.fit_transform(df[all_features])
    feature_names_encoded = prep_mi.get_feature_names_out()

    mi_scores = mutual_info_classif(X_encoded, y_enc, random_state=RANDOM_STATE)
    mi_df = pd.DataFrame({
        "fitur_encoded": feature_names_encoded,
        "MI_score": mi_scores,
    }).sort_values("MI_score", ascending=False)

    # Agregasi MI per fitur asli
    mi_per_fitur = {}
    for fname, score in zip(feature_names_encoded, mi_scores):
        # Cari fitur asli dari nama encoded
        fitur_asli = None
        for col in all_features:
            if fname.startswith(f"cat__{col}_") or fname.startswith(f"num__{col}"):
                fitur_asli = col
                break
        if fitur_asli is None:
            fitur_asli = fname
        if fitur_asli not in mi_per_fitur:
            mi_per_fitur[fitur_asli] = []
        mi_per_fitur[fitur_asli].append(score)

    mi_agg = pd.DataFrame([
        {"fitur": k, "MI_mean": np.mean(v), "MI_max": np.max(v)}
        for k, v in mi_per_fitur.items()
    ]).sort_values("MI_mean", ascending=False)

    print("\nMutual Information per fitur (agregasi):")
    print(mi_agg.to_string(index=False))

    for _, row in mi_agg.iterrows():
        keputusan.append({
            "fitur": row["fitur"], "metode": "filter (MI)",
            "skor": f"MI_mean={row['MI_mean']:.4f}",
            "keputusan": "pertahankan" if row["MI_mean"] > 0.01 else "evaluasi",
            "alasan": "MI tinggi, informatif" if row["MI_mean"] > 0.01
                      else "MI sangat rendah, kontribusi minim",
        })

    # Grafik MI
    fig, ax = plt.subplots(figsize=(10, 6))
    mi_plot = mi_agg.head(15)
    ax.barh(mi_plot["fitur"][::-1], mi_plot["MI_mean"][::-1], color="#7C3AED")
    ax.set_xlabel("Mutual Information (Mean)")
    ax.set_title("Mutual Information per Fitur")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "2_mutual_information.png"), dpi=200)
    plt.close(fig)

    # ==============================================================
    # 3. METODE FILTER: CHI-SQUARE (hanya fitur kategorik)
    # ==============================================================
    print("\n" + "=" * 75)
    print("3. METODE FILTER: CHI-SQUARE")
    print("=" * 75)

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_cat = ohe.fit_transform(df[categorical_cols])
    chi2_scores, chi2_pvals = chi2(X_cat, y_enc)
    cat_feature_names = ohe.get_feature_names_out()

    chi2_agg = {}
    for fname, score, pval in zip(cat_feature_names, chi2_scores, chi2_pvals):
        fitur_asli = None
        for col in categorical_cols:
            if fname.startswith(f"{col}_"):
                fitur_asli = col
                break
        if fitur_asli is None:
            fitur_asli = fname
        if fitur_asli not in chi2_agg:
            chi2_agg[fitur_asli] = {"scores": [], "pvals": []}
        chi2_agg[fitur_asli]["scores"].append(score)
        chi2_agg[fitur_asli]["pvals"].append(pval)

    chi2_df = pd.DataFrame([
        {"fitur": k, "chi2_sum": sum(v["scores"]),
         "chi2_mean": np.mean(v["scores"]),
         "pval_min": min(v["pvals"])}
        for k, v in chi2_agg.items()
    ]).sort_values("chi2_sum", ascending=False)

    print("\nChi-Square per fitur kategorik:")
    print(chi2_df.to_string(index=False))

    for _, row in chi2_df.iterrows():
        keputusan.append({
            "fitur": row["fitur"], "metode": "filter (chi2)",
            "skor": f"chi2_sum={row['chi2_sum']:.2f}, p_min={row['pval_min']:.4f}",
            "keputusan": "pertahankan" if row["pval_min"] < 0.05 else "evaluasi",
            "alasan": "Signifikan secara statistik" if row["pval_min"] < 0.05
                      else "Tidak signifikan (p > 0.05)",
        })

    # ==============================================================
    # 4. METODE WRAPPER: RFE (Recursive Feature Elimination)
    # ==============================================================
    print("\n" + "=" * 75)
    print("4. METODE WRAPPER: RFE")
    print("=" * 75)

    # Gunakan Logistic Regression sebagai estimator RFE
    prep_rfe = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        ('num', StandardScaler(), numerical_cols),
    ])
    X_rfe = prep_rfe.fit_transform(df[all_features])
    rfe_feature_names = prep_rfe.get_feature_names_out()

    estimator = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)

    # RFE dengan kurva skor: coba jumlah fitur dari 1 sampai total
    n_total = X_rfe.shape[1]
    step_sizes = sorted(set(
        [1, 2, 3, 5, 10, 15, 20, 30, 50] +
        list(range(10, min(n_total, 100), 10)) +
        [n_total]
    ))
    step_sizes = [s for s in step_sizes if s <= n_total]

    rfe_scores = []
    for n_feat in step_sizes:
        rfe = RFE(estimator, n_features_to_select=n_feat, step=5)
        rfe.fit(X_rfe, y_enc)
        score = rfe.score(X_rfe, y_enc)
        rfe_scores.append({"n_fitur": n_feat, "accuracy": score})
        if n_feat in [5, 10, 20, n_total]:
            selected = rfe_feature_names[rfe.support_]
            print(f"  n={n_feat:3d}: accuracy={score:.4f} | contoh fitur terpilih: "
                  f"{list(selected[:5])}...")

    rfe_df = pd.DataFrame(rfe_scores)

    # Grafik kurva RFE
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rfe_df["n_fitur"], rfe_df["accuracy"], "o-", color="#DC2626", linewidth=2)
    ax.set_xlabel("Jumlah Fitur")
    ax.set_ylabel("Accuracy (training)")
    ax.set_title("Kurva RFE: Skor vs Jumlah Fitur")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_kurva_rfe.png"), dpi=200)
    plt.close(fig)

    # RFE optimal: ambil n_fitur dimana skor sudah > 95% dari skor maksimal
    max_score = rfe_df["accuracy"].max()
    threshold_95 = max_score * 0.95
    optimal_rfe = rfe_df[rfe_df["accuracy"] >= threshold_95].iloc[0]
    print(f"\n  RFE optimal: {int(optimal_rfe['n_fitur'])} fitur "
          f"(accuracy {optimal_rfe['accuracy']:.4f}, threshold 95% = {threshold_95:.4f})")

    # Catat fitur terpilih RFE
    rfe_final = RFE(estimator, n_features_to_select=int(optimal_rfe['n_fitur']), step=5)
    rfe_final.fit(X_rfe, y_enc)
    selected_rfe = set(rfe_feature_names[rfe_final.support_])

    # Map kembali ke fitur asli
    for col in all_features:
        terpilih = any(f.startswith(f"cat__{col}_") or f.startswith(f"num__{col}")
                       for f in selected_rfe)
        keputusan.append({
            "fitur": col, "metode": "wrapper (RFE)",
            "skor": f"terpilih={terpilih}",
            "keputusan": "pertahankan" if terpilih else "buang",
            "alasan": "Terpilih oleh RFE" if terpilih else "Dieliminasi oleh RFE",
        })

    # ==============================================================
    # 5. METODE EMBEDDED: RANDOM FOREST IMPORTANCE
    # ==============================================================
    print("\n" + "=" * 75)
    print("5. METODE EMBEDDED: RANDOM FOREST FEATURE IMPORTANCE")
    print("=" * 75)

    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_rfe, y_enc)
    rf_imp = pd.DataFrame({
        "fitur_encoded": rfe_feature_names,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\nTop 15 fitur (RF importance):")
    print(rf_imp.head(15).to_string(index=False))

    # Agregasi per fitur asli
    rf_agg = {}
    for fname, imp in zip(rfe_feature_names, rf.feature_importances_):
        fitur_asli = None
        for col in all_features:
            if fname.startswith(f"cat__{col}_") or fname.startswith(f"num__{col}"):
                fitur_asli = col
                break
        if fitur_asli is None:
            fitur_asli = fname
        rf_agg[fitur_asli] = rf_agg.get(fitur_asli, 0) + imp

    rf_agg_df = pd.DataFrame([
        {"fitur": k, "importance_sum": v} for k, v in rf_agg.items()
    ]).sort_values("importance_sum", ascending=False)
    print("\nImportance per fitur asli (RF, agregasi):")
    print(rf_agg_df.to_string(index=False))

    for _, row in rf_agg_df.iterrows():
        keputusan.append({
            "fitur": row["fitur"], "metode": "embedded (RF)",
            "skor": f"importance={row['importance_sum']:.4f}",
            "keputusan": "pertahankan" if row["importance_sum"] > 0.01 else "evaluasi",
            "alasan": "Kontribusi penting menurut RF" if row["importance_sum"] > 0.01
                      else "Kontribusi sangat kecil menurut RF",
        })

    # Grafik RF importance
    fig, ax = plt.subplots(figsize=(10, 6))
    top = rf_agg_df.head(15)
    ax.barh(top["fitur"][::-1], top["importance_sum"][::-1], color="#0D9488")
    ax.set_xlabel("Total Feature Importance")
    ax.set_title("Random Forest Feature Importance (Agregasi per Fitur Asli)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "4_rf_importance.png"), dpi=200)
    plt.close(fig)

    # ==============================================================
    # 6. METODE EMBEDDED: KOEFISIEN LASSO
    # ==============================================================
    print("\n" + "=" * 75)
    print("6. METODE EMBEDDED: KOEFISIEN LASSO")
    print("=" * 75)

    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(df[numerical_cols])
    # Lasso pada fitur numerikal saja (target: encoded label)
    lasso = Lasso(alpha=0.01, max_iter=5000, random_state=RANDOM_STATE)
    lasso.fit(X_num_scaled, y_enc)

    lasso_df = pd.DataFrame({
        "fitur": numerical_cols,
        "koefisien_lasso": lasso.coef_,
        "abs_koefisien": np.abs(lasso.coef_),
    }).sort_values("abs_koefisien", ascending=False)

    print("\nKoefisien Lasso (numerikal):")
    print(lasso_df.to_string(index=False))

    for _, row in lasso_df.iterrows():
        keputusan.append({
            "fitur": row["fitur"], "metode": "embedded (Lasso)",
            "skor": f"koef={row['koefisien_lasso']:.4f}",
            "keputusan": "pertahankan" if row["abs_koefisien"] > 0.001 else "buang",
            "alasan": "Koefisien non-zero (Lasso tidak menyusutkan ke nol)"
                      if row["abs_koefisien"] > 0.001
                      else "Koefisien mendekati nol → Lasso menganggap tidak penting",
        })

    # ==============================================================
    # 7. ANALISIS TF-IDF PADA DESKRIPSI
    # ==============================================================
    print("\n" + "=" * 75)
    print("7. ANALISIS TF-IDF PADA DESKRIPSI")
    print("=" * 75)

    deskripsi_bersih = df['deskripsi_bersih'].fillna("").astype(str)
    tfidf = TfidfVectorizer(max_features=500, min_df=5, max_df=0.8)
    X_tfidf = tfidf.fit_transform(deskripsi_bersih)
    vocab = tfidf.get_feature_names_out()

    # Rata-rata TF-IDF per label → kata yang paling membedakan
    print("\nKata TF-IDF teratas per label:")
    tfidf_per_label = {}
    for label in LABEL_ORDER:
        mask = (y == label).values
        if mask.sum() == 0:
            continue
        mean_tfidf = np.asarray(X_tfidf[mask].mean(axis=0)).flatten()
        top_idx = mean_tfidf.argsort()[-10:][::-1]
        top_words = [(vocab[i], mean_tfidf[i]) for i in top_idx]
        tfidf_per_label[label] = top_words
        print(f"\n  [{label}]:")
        for word, score in top_words:
            print(f"    {word:20s} TF-IDF={score:.4f}")

    # Kata yang paling MEMBEDAKAN antar label (selisih TF-IDF Cepat vs Lambat)
    if "Cepat" in tfidf_per_label and "Lambat" in tfidf_per_label:
        mask_cepat = (y == "Cepat").values
        mask_lambat = (y == "Lambat").values
        mean_c = np.asarray(X_tfidf[mask_cepat].mean(axis=0)).flatten()
        mean_l = np.asarray(X_tfidf[mask_lambat].mean(axis=0)).flatten()
        diff = mean_c - mean_l
        top_cepat_idx = diff.argsort()[-10:][::-1]
        top_lambat_idx = diff.argsort()[:10]

        print("\n  Kata lebih sering di CEPAT (vs Lambat):")
        for i in top_cepat_idx:
            print(f"    {vocab[i]:20s} diff={diff[i]:+.4f}")
        print("\n  Kata lebih sering di LAMBAT (vs Cepat):")
        for i in top_lambat_idx:
            print(f"    {vocab[i]:20s} diff={diff[i]:+.4f}")

    keputusan.append({
        "fitur": "deskripsi_bersih (TF-IDF)", "metode": "analisis TF-IDF",
        "skor": f"vocab_size={len(vocab)}",
        "keputusan": "referensi",
        "alasan": "Analisis TF-IDF menunjukkan kata-kata yang membedakan label. "
                  "Fitur teks (ada_garansi, riwayat_negatif, dll.) sudah mengekstrak "
                  "informasi utama dari deskripsi. TF-IDF penuh tidak dimasukkan sbg "
                  "fitur classifier karena dimensi tinggi dan risiko overfitting.",
    })

    # ==============================================================
    # RINGKASAN KEPUTUSAN FITUR
    # ==============================================================
    print("\n" + "=" * 75)
    print("RINGKASAN KEPUTUSAN FITUR")
    print("=" * 75)

    df_keputusan = pd.DataFrame(keputusan)
    df_keputusan.to_csv(KEPUTUSAN_FILE, index=False, encoding='utf-8-sig')
    print(f"\nKeputusan fitur disimpan ke '{KEPUTUSAN_FILE}'.")

    # Ringkasan per fitur asli: kesimpulan dari semua metode
    print("\nRingkasan per fitur:")
    for col in all_features + ['deskripsi_bersih (TF-IDF)']:
        entries = df_keputusan[df_keputusan['fitur'] == col]
        if entries.empty:
            continue
        buang_count = (entries['keputusan'] == 'buang').sum()
        total = len(entries)
        verdict = "BUANG" if buang_count > total / 2 else "PERTAHANKAN"
        print(f"  {col:30s}: {verdict:12s} "
              f"(buang {buang_count}/{total} metode)")

    # Keputusan akhir: SATU-SATUNYA daftar fitur yang dibaca train_model_klasifikasi.py
    fitur_redundan = {c2: f"redundan dengan {c1} (r={abs(r):.4f})" for c1, c2, r in redundan}
    alasan_buang = {}
    for c in all_features:
        if df[c].nunique() <= 1:
            alasan_buang[c] = "konstan (hanya satu nilai, tidak membawa informasi)"
        elif c in fitur_redundan:
            alasan_buang[c] = fitur_redundan[c]
    fitur_dipertahankan = [c for c in all_features if c not in alasan_buang]
    baris_akhir = [{"fitur": c, "metode": "KEPUTUSAN AKHIR",
                    "keputusan": "buang" if c in alasan_buang else "pertahankan",
                    "alasan": alasan_buang.get(c, "lolos: tidak konstan dan tidak redundan")}
                   for c in all_features]
    df_keputusan = pd.concat([df_keputusan, pd.DataFrame(baris_akhir)], ignore_index=True)
    df_keputusan.to_csv(KEPUTUSAN_FILE, index=False, encoding='utf-8-sig')
    with open("fitur_final.json", "w", encoding="utf-8") as f_json:
        json.dump({"dipertahankan": fitur_dipertahankan, "dibuang": alasan_buang}, f_json,
                  indent=2, ensure_ascii=False)
    print(f"\n  KEPUTUSAN AKHIR -> dipertahankan: {fitur_dipertahankan}")
    for c, a in alasan_buang.items():
        print(f"                     dibuang     : {c}  ({a})")
    print(f"  Disimpan di fitur_final.json (dibaca train_model_klasifikasi.py) dan {KEPUTUSAN_FILE}.")

    # Grafik ringkasan
    fig, ax = plt.subplots(figsize=(12, 6))
    fitur_list = list(rf_agg_df["fitur"])
    mi_vals = [mi_per_fitur.get(f, [0]) for f in fitur_list]
    mi_means = [np.mean(v) for v in mi_vals]
    rf_vals = [rf_agg.get(f, 0) for f in fitur_list]

    x = np.arange(len(fitur_list))
    ax.bar(x - 0.2, mi_means, 0.4, label="MI (mean)", color="#7C3AED")
    ax.bar(x + 0.2, rf_vals, 0.4, label="RF Importance", color="#059669")
    ax.set_xticks(x)
    ax.set_xticklabels(fitur_list, rotation=45, ha="right", fontsize=9)
    ax.set_title("Perbandingan Mutual Information vs RF Importance")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "5_ringkasan_fitur.png"), dpi=200)
    plt.close(fig)

    print(f"\n  Grafik disimpan di '{OUTPUT_DIR}/'.")
    print("\n[SELESAI] Analisis fitur selesai. Lanjutkan ke train_model_klasifikasi.py.")


if __name__ == "__main__":
    main()