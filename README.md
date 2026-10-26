# 🚗 AutoLaku: Sistem Prediksi Kecepatan Laku Mobil Bekas OLX Berdasarkan Harga Wajar, Jarak Tempuh, dan Deskripsi Iklan Menggunakan Metode Klasifikasi

---

## 📌 Ringkasan Proyek & Proposal

| Komponen | Keterangan / Spesifikasi |
|---|---|
| **Nama Produk** | **AutoLaku** |
| **Masalah Utama** | Penjual mobil bekas perseorangan di OLX kerap mengalami *asimetri informasi* dan kesulitan menentukan strategi harga jual. Memasang harga terlalu mahal (*overpriced*) berisiko membuat kendaraan mengendap lama di garasi (*liquidity risk*), sedangkan harga terlalu murah berisiko merugikan penjual. Selain itu, penjual sering tidak menyadari bahwa faktor kualitatif dalam deskripsi (seperti garansi/catatan servis) dan rasio kilometer tahunan berpengaruh besar terhadap daya serap pasar. |
| **Profil Pengguna Target** | **Penjual Mobil Bekas Perseorangan** yang ingin mengevaluasi daya saing iklan, memprediksi kecepatan laku unitnya (Cepat, Sedang, Lambat), serta mensimulasikan penyesuaian harga sebelum iklan resmi ditayangkan. |
| **Sumber Dataset** | Web scraping mandiri via API Interception & Playwright pada direktori mobil bekas OLX Indonesia (`dataset_olx_mentah.csv`), menghasilkan **3.000 baris data iklan aktif**. |
| **Fungsi Utama Aplikasi** | Menerima input spesifikasi teknis kendaraan, rencana harga jual, dan draf deskripsi iklan. Sistem menghitung estimasi **Harga Pasar Wajar (ML Regression)**, memicu **NLP Text Scanner**, dan memprediksi status **Kecepatan Laku (Classification)** beserta simulasi harga interaktif. |
| **Metodologi Data Mining** | **Classification** (Prediksi kategori kecepatan laku: *Cepat*, *Sedang*, *Lambat*) dengan **Regression** (Gradient Boosting / Log-Price) sebagai *feature engine* pembentuk fitur utama `deviasi_persen`. |

---

## 🏗️ Pipeline Sistem End-to-End

```
┌─────────────────────────────────────────────────────────────────┐
│  TAHAP 0 — SCRAPING  (scraping.py)                              │
│                                                                 │
│  • Buka halaman pencarian mobil bekas di OLX via Playwright     │
│  • Intersepsi respons API JSON internal OLX                     │
│  • Ambil atribut: judul, merek, model, tahun, transmisi,        │
│    harga, jarak_tempuh, tipe_penjual, deskripsi, tanggal_posting│
│                                                                 │
│  Output ➔ dataset_olx_mentah.csv                                │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  TAHAP 1 — PREPROCESSING & FEATURE ENGINEERING (processing.py)  │
│                                                                 │
│  • Ekstraksi merek & model spesifik dari judul iklan            │
│  • Filter harga boneka/placeholder (99999999, 88888888, dll.)    │
│  • Imputasi jarak tempuh rentang/kosong dengan median tahunan   │
│  • Hitung usia_mobil & km_per_tahun                             │
│  • NLP Teks Scanner: Deteksi garansi & TDP kredit (dengan negasi)│
│  • Hitung usia_bracket & jumlah_sampel_kategori                 │
│                                                                 │
│  Output ➔ dataset_olx_bersih.csv                                │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  TAHAP 2 — PELATIHAN MODEL & EVALUASI (train_model_klasifikasi) │
│                                                                 │
│  [2a] Model Regresi Harga Wajar (Log-Price Gradient Boosting)   │
│       • Hitung harga_wajar_model secara Out-of-Fold (OOF 5-Fold)│
│       • Fallback Regresi Usia/Median jika sampel < 15 data      │
│  [2b] Pembentukan Label Proxy Likuiditas                        │
│       • deviasi_persen = (harga_iklan - harga_wajar) / harga_wajar│
│       • Pembagian Tertile (Q33 & Q67): Cepat / Sedang / Lambat  │
│       • Penyesuaian Rule: KM/tahun > 30.000 & Tanpa Garansi ➔ Sedang│
│  [2c] Model Klasifikasi Kecepatan Laku (Stratified 5-Fold CV)   │
│       • Pengujian Kandidat: LogReg, DecisionTree, RF, GradBoost │
│       • Hyperparameter Tuning (GridSearchCV)                    │
│       • Kriteria Pilih: Monotonik Harga, Non-Overconfident, F1   │
│                                                                 │
│  Output ➔ model_regresi_harga.pkl, model_klasifikasi_likuiditas.pkl,│
│           autolaku_meta.json, folder output_klasifikasi_<date>/ │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  TAHAP 3 — CORE INFERENCE ENGINE (mesin_prediksi.py)            │
│                                                                 │
│  • Pusat logika inferensi tunggal: validasi masukan, hitung     │
│    harga wajar, hitung deviasi, prediksi probabilitas kelas,    │
│    dan penyusunan peringatan keyakinan sampel.                  │
└─────────────────────────────────────────────────────────────────┘
     │
     ├───────────────────────────────┐
     ▼                               ▼
┌──────────────────────────┐    ┌─────────────────────────────────┐
│ TAHAP 4 — AUTOMATED TEST │    │ TAHAP 5 — USER INTERFACE        │
│ (test_sistem.py)         │    │                                 │
│                          │    │ • CLI Terminal:                 │
│ • Validasi masukan & NLP │    │   test_prediksi_klasifikasi.py  │
│ • Uji Monotonisitas Harga│    │ • Web App (Streamlit):          │
└──────────────────────────┘    │   app.py                        │
                                └─────────────────────────────────┘
```

---

## 📊 Atribut Sistem Lengkap

### 1. Atribut Mentah (Dataset OLX - 10 Kolom)

| Kolom | Tipe Data | Deskripsi & Fungsi |
|---|---|---|
| `judul` | Teks | Judul utama iklan. Digunakan untuk fallback ekstraksi model, perbaikan typo, dan verifikasi fitur NLP. |
| `merek` | Kategorik | Merek pabrikan kendaraan (Toyota, Honda, Daihatsu, Suzuki, Mitsubishi, dll.). |
| `model` | Kategorik | Seri/Model spesifik kendaraan (Avanza, Innova, Brio, Xenia, Xpander, dll.). |
| `tahun` | Numerik | Tahun perakitan kendaraan (1995–2026). Dikonversi dari kolom atau diekstrak dari judul. |
| `transmisi` | Kategorik | Jenis sistem transmisi, dinormalisasi menjadi dua kelas baku: `automatic` atau `manual`. |
| `harga` | Numerik | Harga penawaran aktif penjual (Rupiah). Filter harga boneka/placeholder diterapkan. |
| `jarak_tempuh` | Numerik | Total angka odometer (KM). Format rentang (mis. "50.000–60.000") dirata-ratakan. |
| `deskripsi` | Teks | Teks penjelasan detail iklan. Dipindai oleh NLP engine untuk fitur garansi dan indikasi kredit. |
| `tipe_penjual_badge` | Kategorik | Lencana penjual di OLX, dinormalisasi menjadi `Individu` atau `Dealer`. |
| `tanggal_posting` | Teks | Tanggal tayang iklan di platform OLX (digunakan untuk validasi kasar umur iklan). |

### 2. Atribut Turunan (*Feature Engineering* - 7 Kolom Baru)

| Kolom | Dibuat Dari | Penjelasan Metodologis |
|---|---|---|
| `usia_mobil` | `tahun` | $2026 - \text{tahun}$. Penentu utama kurva depresiasi nilai kendaraan. |
| `km_per_tahun` | `jarak_tempuh` ÷ `usia_mobil` | Rata-rata pemakaian tahunan. Mengukur tingkat intensitas operasional kendaraan. |
| `ada_garansi` | `deskripsi`, `judul` | Biner (`Ya`/`Tidak`). "Ya" jika terdapat kata kunci garansi/servis/bebas banjir **tanpa** negasi. |
| `indikasi_kredit_tdp` | `deskripsi`, `judul` | Biner (`1`/`0`). 1 jika deskripsi menyebut kata TDP, DP minim, cicilan, atau over kredit. |
| `tipe_penjual` | `tipe_penjual_badge` | Normalisasi tipe akun penjual (`Individu` / `Dealer`). |
| `usia_bracket` | `usia_mobil` | Pengelompokan umur kendaraan per rentang 5 tahun (0–4, 5–9, 10–14, dst.). |
| `jumlah_sampel_kategori` | `merek`, `model`, `usia_bracket` | Jumlah unit sejenis dalam kancah data. Menjadi dasar peringatan keyakinan sampel (ambang = 15). |

### 3. Matriks Fitur Pemodelan

* **Model Regresi Harga Wajar (9 Fitur):** `merek` [One-Hot], `model` [One-Hot], `transmisi` [One-Hot], `tipe_penjual` [One-Hot], `ada_garansi` [One-Hot], `tahun` [StandardScaler], `jarak_tempuh` [StandardScaler], `usia_mobil` [StandardScaler], `km_per_tahun` [StandardScaler].
* **Model Klasifikasi Kecepatan Laku (10 Fitur):** 9 Fitur Regresi di atas + `deviasi_persen` (Fitur Terpenting).

---

## 🏷️ Pembentukan Label Proxy & Aturan Bisnis

Karena OLX menyembunyikan iklan yang telah terjual (*unlisted*), label target (**Cepat**, **Sedang**, **Lambat**) dibentuk secara kuantitatif (*Proxy Labeling*):

1. **Estimasi Out-of-Fold (OOF):** Harga wajar tiap iklan dihitung menggunakan 5-Fold Cross-Validation agar tidak terjadi *data leakage*.
2. **Kalkulasi Deviasi Persen:**
   $$\text{deviasi\_persen} = \frac{\text{harga\_iklan} - \text{harga\_wajar}}{\text{harga\_wajar}} \times 100\%$$
3. **Pembagian Tertile Kuantil:**
   - **Q33 ($\approx -8,07\%$):** Deviasi $\le -8,07\%$ ➔ **Cepat Laku** ($< 15$ Hari)
   - **Q67 ($\approx +7,67\%$):** Deviasi $-8,07\%$ s.d. $+7,67\%$ ➔ **Laku Sedang** ($15 - 30$ Hari)
   - **$>$ Q67:** Deviasi $> +7,67\%$ ➔ **Lambat Laku** ($> 30$ Hari)
4. **Aturan Penyesuaian Kondisi Fisik:** Unit di kelas *Cepat* dengan `km_per_tahun` $> 30.000$ KM **dan** `ada_garansi` = `Tidak` otomatis diturunkan kelasnya menjadi **Sedang**.

---

## 🤖 Perbandingan Model & Metrik Evaluasi Akademik (P12 & P13)

### 1. Evaluasi Model Regresi Harga Wajar

| Metrik Evaluasi | Nilai Capaian | Keterangan Akademik |
|---|---|---|
| **MAPE (Mean Absolute Percentage Error)** | **19,49%** | Rata-rata persentase galat prediksi harga wajar terhadap pasaran. |
| **MAE (Mean Absolute Error)** | **Rp 81.606.098** | Rata-rata selisih mutlak nominal harga prediksi. |
| **RMSE (Root Mean Squared Error)** | **Rp 197.722.509** | Tingkat sensitivitas galat terhadap nilai ekstrem/mobil mewah. |

### 2. Perbandingan Model Klasifikasi (Materi P13)

| Algoritma Klasifikasi | Accuracy | Precision (Macro) | Recall (Macro) | F1-Macro | 5-Fold CV F1 (Mean ± Std) | Monotonisitas Harga | Overconfidence Check |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 98,25% | 98,30% | 98,20% | 98,25% | $0,9812 \pm 0,0042$ | **Lolos (100%)** | Lolos ($< 0,97$) |
| **Decision Tree** | 98,80% | 98,82% | 98,78% | 98,80% | $0,9865 \pm 0,0031$ | **Lolos (100%)** | Gagal (Overconfident 1,00) |
| **Random Forest (Tuned)** | **99,14%** | **99,15%** | **99,12%** | **99,14%** | **$0,9908 \pm 0,0021$** | **Lolos (100%)** | **Lolos (0,692)** |
| **Gradient Boosting** | 99,05% | 99,06% | 99,04% | 99,05% | $0,9898 \pm 0,0025$ | **Lolos (100%)** | Lolos (0,885) |

* **Model Terpilih:** **Random Forest (Tuned)** dipilih karena memiliki *Mean F1-Score* tertinggi pada 5-Fold Cross Validation, lulus uji monotonisitas harga 100%, dan memiliki tingkat keyakinan puncak yang realistis ($0,692$).

### 3. Hasil Uji Ablation (Sirkularitas Label)

| Skenario Pengujian Fitur | Accuracy | F1-Score (Macro) | Selisih Impak |
|---|---|---|---|
| **Model Lengkap (Dengan `deviasi_persen`)** | **99,14%** | **99,14%** | Baseline |
| **Model Ablation (Tanpa `deviasi_persen`)** | **53,75%** | **53,10%** | **-45,39 Poin** |

* **Analisis Akademik:** Penurunan akurasi sebesar 45,4% membuktikan bahwa fitur `deviasi_persen` mendominasi keputusan klasifikasi, menegaskan bahwa label dibentuk dari deviasi harga wajar.

---

## 📂 Struktur File Repositori Bersih

| Nama Berkas / Folder | Deskripsi & Peran Sistem |
|---|---|
| `scraping.py` | Skrip web scraping otomatis berbasis Playwright untuk mengunduh data iklan OLX. |
| `kamus_slang_otomotif.py` | Modul kamus normalisasi untuk mengganti istilah slang, singkatan, dan typo otomotif. |
| `processing.py` | Pipeline pembersihan data, penanganan harga boneka, ekstraksi fitur NLP & turunan. |
| `train_model_klasifikasi.py` | Skrip pelatihan model regresi harga wajar & klasifikasi 5-Fold Cross Validation. |
| `mesin_prediksi.py` | Engine inferensi terpusat yang memuat model `.pkl`, validasi input, dan kalkulasi prediksi. |
| `test_prediksi_klasifikasi.py` | Antarmuka berbasis Terminal CLI interaktif khusus untuk persona Penjual. |
| `test_sistem.py` | Pengujian unit otomatis (validasi input, NLP negasi, dan uji monotonisitas harga). |
| `app.py` | Dashboard aplikasi web interaktif menggunakan Streamlit (Persona Penjual & Pembeli). |
| `dataset_olx_mentah.csv` | Dataset mentah hasil scraping dari platform OLX Indonesia (3.000 baris). |
| `dataset_olx_bersih.csv` | Dataset bersih hasil preprocessing dan feature engineering. |
| `model_regresi_harga.pkl` | Binary serialisasi model regresi Log-Price Gradient Boosting Regressor. |
| `model_klasifikasi_likuiditas.pkl` | Binary serialisasi model klasifikasi Random Forest Classifier. |
| `autolaku_meta.json` | Berkas konfigural metadata model (ambang tertile, MAPE, metrik evaluasi). |
| `output_klasifikasi_<tanggal>/` | Folder output otomatis berisi 9 grafik evaluasi visual standar P12/P13. |

---

## 🚀 Panduan Eksekusi Lengkap

Jalankan perintah berikut secara berurutan pada terminal:

```powershell
# 1. Instalasi Dependensi Pustaka
pip install pandas numpy scikit-learn matplotlib scipy streamlit playwright

# 2. Pembersihan & Preprocessing Data Mentah
python processing.py

# 3. Pelatihan Model, Cross-Validation, & Generasi Grafik Evaluasi
python train_model_klasifikasi.py

# 4. Eksplorasi Pengujian Otomatis Bebas Bug
python test_sistem.py

# 5. Jalankan Antarmuka Terminal (CLI)
python test_prediksi_klasifikasi.py

# 6. Jalankan Aplikasi Web Streamlit Interactive Dashboard
streamlit run app.py
```

---

## ⚠️ Keterbatasan Sistem & Kejujuran Akademik

1. **Sifat Proxy Labeling:** Ketiadaan data historis tanggal transaksi cair di OLX membuat label *Cepat/Sedang/Lambat* murni dibentuk dari simpangan harga wajar.
2. **Sirkularitas Fitur:** Fitur `deviasi_persen` memegang peranan 45,4% terhadap akurasi model. Akurasi 99,1% menunjukkan model berhasil menghafal pola pembagian tertile harga wajar.
3. **Peringatan Sample Scarcity:** Untuk kombinasi merek, model, dan usia langka ($< 15$ data), sistem secara otomatis mengalihkan perhitungan ke metode *Fallback Log-Age Regression* dan menyalakan peringatan keyakinan rendah pada antarmuka pengguna.