# 🚗 AutoLaku: Sistem Prediksi Kecepatan Laku Mobil Bekas OLX Berdasarkan Harga Wajar, Jarak Tempuh, dan Deskripsi Iklan Menggunakan Metode Klasifikasi

---

## 📌 Ringkasan Proyek & Proposal

| Komponen | Keterangan / Spesifikasi |
|---|---|
| **Nama Produk** | **AutoLaku** |
| **Masalah Utama** | Penjual mobil bekas perseorangan di platform OLX kerap mengalami *asimetri informasi* dan kesulitan menentukan strategi harga jual. Memasang harga terlalu mahal (*overpriced*) berisiko membuat kendaraan mengendap lama di garasi (*liquidity risk*), sedangkan harga terlalu murah berisiko merugikan penjual. Selain itu, penjual sering tidak menyadari bahwa faktor kualitatif dalam deskripsi (seperti garansi/catatan servis) dan rasio kilometer tahunan berpengaruh besar terhadap daya serap pasar. |
| **Profil Pengguna Target** | **Penjual Mobil Bekas Perseorangan** yang ingin mengevaluasi daya saing iklan, memprediksi kecepatan laku unitnya (Cepat, Sedang, Lambat), serta mensimulasikan penyesuaian harga sebelum iklan resmi ditayangkan. |
| **Sumber Dataset** | Web scraping mandiri via API Interception & Playwright pada direktori mobil bekas OLX Indonesia (`dataset_olx_mentah.csv`), menghasilkan **1.000 baris data iklan aktif bersih**. |
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
│  TAHAP 2 — MODEL HARGA WAJAR & PELABELAN PROXY (pelabelan.py)   │
│                                                                 │
│  • Data Split: 80% Train (800 sampel) & 20% Test (200 sampel)   │
│  • Regresi Log-Target Gradient Boosting: MAE Rp 98,3 Juta       │
│  • Kalkulasi Deviasi Persen pada Data Training                  │
│  • Ambang Tertile: Q33 (-18,71%) & Q67 (+34,62%)               │
│                                                                 │
│  Output ➔ dataset_olx_berlabel.csv                              │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  TAHAP 3 — FEATURE SELECTION FORMAL (analisis_fitur.py)        │
│                                                                 │
│  • Pearson Correlation: Deteksi Multikolinearitas (r = -1.0)    │
│  • Dibuang: `tahun` (Redundan dengan `usia_mobil`)              │
│  • Dipertahankan: `usia_mobil`, `jarak_tempuh`, `km_per_tahun`, │
│    `merek`, `model`, `transmisi`, `ada_garansi`, `tdp_kredit`   │
│                                                                 │
│  Output ➔ keputusan_fitur.csv, analisis_fitur_meta.json         │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  TAHAP 4 — PELATIHAN & EVALUASI KLASIFIKASI (train_model.py)    │
│                                                                 │
│  • 5-Fold Stratified Cross-Validation pada Data Training        │
│  • Pengujian 6 Algoritma (LogReg, KNN, NB, DT, RF, GradBoost)   │
│  • Skenario A (Ablation Study) vs Skenario B (Model Utama)     │
│  • Generasi 6 Grafik Visualisasi Akademis (%)                   │
│                                                                 │
│  Output ➔ model_klasifikasi_likuiditas.pkl,                     │
│           ringkasan_evaluasi_klasifikasi.json,                  │
│           folder output_klasifikasi_terbaru/                    │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  TAHAP 5 — INFERENCE ENGINE & INTERFACE (app.py)                │
│                                                                 │
│  • Core Engine: mesin_prediksi.py                               │
│  • Interactive Dashboard: Streamlit Web App (app.py)            │
└─────────────────────────────────────────────────────────────────┘
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
| `transmisi` | Kategorik | Jenis sistem transmisi, dinormalisasi menjadi dua kelas baku: `Automatic` atau `Manual`. |
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

### 3. Matriks Fitur Pemodelan Final (`keputusan_fitur.csv`)

| Fitur | Tipe Data | Status | Alasan Akademik & Metodologis |
|---|---|---|---|
| `tahun` | Numerik | **DIBUANG** | Multikolinearitas sempurna ($r = -1.0$) dengan `usia_mobil`. |
| `usia_mobil` | Numerik | **DIPERTAHANKAN** | Fitur linier depresiasi harga kendaraan per tahun. |
| `jarak_tempuh` | Numerik | **DIPERTAHANKAN** | Indikator utama pemakaian fisik kendaraan. |
| `km_per_tahun` | Numerik | **DIPERTAHANKAN** | Rasio intensitas pemakaian operasional tahunan. |
| `merek` & `model` | Kategorik | **DIPERTAHANKAN** | Variabel utama segmentasi pasar. |
| `transmisi` | Kategorik | **DIPERTAHANKAN** | Preferensi utama konsumen pasar lokal. |
| `tipe_penjual` | Kategorik | **DIPERTAHANKAN** | Memisahkan perilaku harga Dealer vs Individu. |
| `ada_garansi` | Kategorik | **DIPERTAHANKAN** | Fitur NLP penambah nilai psikologis pembeli. |
| `indikasi_kredit_tdp`| Biner | **DIPERTAHANKAN** | Menyaring skema penawaran uang muka kredit. |
| `deviasi_persen` | Numerik | **DIPERTAHANKAN** | Fitur utama selisih harga jual terhadap harga wajar pasar. |

---

## 🏷️ Pembentukan Label Proxy & Aturan Bisnis

Karena OLX menyembunyikan iklan yang telah terjual (*unlisted*), label target (**Cepat**, **Sedang**, **Lambat**) dibentuk secara kuantitatif (*Proxy Labeling*) pada **Data Training ($800$ sampel)**:

1. **Estimasi Regresi Harga Wajar:** Model Log-Target Gradient Boosting dilatih pada data training.
2. **Kalkulasi Deviasi Persen:**
   $$\text{deviasi\_persen} = \frac{\text{harga\_iklan} - \text{harga\_wajar}}{\text{harga\_wajar}} \times 100\%$$
3. **Pembagian Tertile Kuantil (Data Train):**
   - **$Q_{33}$ (Batas $-18,71\%$):** Deviasi $\le -18,71\%$ ➔ **Cepat Laku**
   - **$Q_{67}$ (Batas $+34,62\%$):** Deviasi $-18,71\%$ s.d. $+34,62\%$ ➔ **Sedang Laku**
   - **$>$ $Q_{67}$:** Deviasi $> +34,62\%$ ➔ **Lambat Laku**

---

## 🤖 Perbandingan Model & Metrik Evaluasi Akademik (P6, P9, P12, P13)

### 1. Skenario A: Ablation Study (Murni Spesifikasi Tanpa Deviasi Harga)

| Model | Kategori Algoritma | CV Val F1 (%) | Test F1 (%) | Test Acc (%) |
|---|---|---|---|---|
| **Logistic Regression** | P9 (Linear Model) | 29,62% | 35,35% | 36,50% |
| **K-Nearest Neighbors (KNN)**| P6 (Instance-Based) | 31,19% | 32,78% | 33,00% |
| **Naive Bayes (Gaussian)** | P6 (Probabilistic) | 30,02% | 42,67% | 42,50% |
| **Decision Tree** | P12 (Tree Model) | 29,93% | 20,86% | 30,50% |
| **Random Forest** | P13 (Bagging Ensemble) | 29,16% | 31,17% | 31,50% |
| **Gradient Boosting** | P13 (Boosting Ensemble) | 33,16% | 30,62% | 31,00% |

> **Analisis Akademis:** Tanpa variabel deviasi harga, skor F1-Score seluruh algoritma bernilai rendah ($\approx 30\%$). Ini membuktikan bahwa spesifikasi fisik mobil saja tidak cukup untuk memprediksi Kecepatan Laku, dan menegaskan bahwa deviasi harga pasar adalah penentu utama likuiditas.

### 2. Skenario B: Model Utama Aplikasi (Dengan Fitur Deviasi Harga)

| Algoritma Klasifikasi | CV Train F1 (%) | CV Val F1 (%) | CV Std (%) | Test Acc (%) | Test Prec (%) | Test Rec (%) | Test F1 (%) |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 96,30% | 92,44% | $\pm 1,66\%$ | 95,00% | 94,96% | 95,08% | 94,98% |
| **K-Nearest Neighbors (KNN)**| 82,93% | 69,83% | $\pm 4,76\%$ | 71,00% | 71,75% | 71,13% | 71,40% |
| **Naive Bayes (Gaussian)** | 78,66% | 71,63% | $\pm 4,28\%$ | 78,50% | 78,35% | 78,94% | 78,48% |
| **Decision Tree (Terpilih)**| **100,00%** | **100,00%** | **$\pm 0,00\%$** | **100,00%** | **100,00%** | **100,00%** | **100,00%** |
| **Random Forest** | 100,00% | 99,25% | $\pm 0,25\%$ | 98,50% | 98,48% | 98,57% | 98,49% |
| **Gradient Boosting** | 100,00% | 100,00% | $\pm 0,00\%$ | 100,00% | 100,00% | 100,00% | 100,00% |

* **Model Terpilih:** **Decision Tree** dipilih sebagai model utama aplikasi karena mencapai F1-Score 100,00% dengan tingkat stabilitas Cross-Validation yang sempurna ($\pm 0,00\%$).

---

## 📂 Struktur File Repositori Bersih

| Nama Berkas / Folder | Deskripsi & Peran Sistem |
|---|---|
| `processing.py` | Script preprocessing data mentah, imputasi, dan ekstraksi fitur NLP (Materi P3-4). |
| `pelabelan.py` | Script regresi harga wajar & pembentukan proxy label tertile (Materi P5 & P12). |
| `analisis_fitur.py` | Script seleksi fitur formal Pearson & Mutual Information (Materi P3-4). |
| `train_model_klasifikasi.py` | Script pelatihan 6 model klasifikasi & pembuat 6 grafik visualisasi (Materi P6, P9, P12, P13). |
| `mesin_prediksi.py` | Core inference engine terpusat pemuat file `.pkl` dan validator input. |
| `app.py` | Dashboard aplikasi web interaktif Streamlit untuk persona penjual. |
| `dataset_olx_bersih.csv` | Dataset hasil pembersihan tahap preprocessing. |
| `dataset_olx_berlabel.csv` | Dataset berlabel dengan penanda isolasi `is_train`. |
| `keputusan_fitur.csv` | Tabel formulasi seleksi fitur formal. |
| `model_klasifikasi_likuiditas.pkl` | Binary serialisasi pipeline model klasifikasi terpilih. |
| `output_klasifikasi_terbaru/` | Folder output berisi 6 gambar grafik visualisasi evaluasi akademis. |

---

## 🚀 Panduan Eksekusi Sistem

Jalankan perintah berikut secara berurutan pada terminal PowerShell / VS Code:

```powershell
# 1. Tahap Preprocessing Data Mentah
py processing.py

# 2. Tahap Pelabelan Proxy & Model Regresi Harga
py pelabelan.py

# 3. Tahap Seleksi Fitur Formal
py analisis_fitur.py

# 4. Tahap Pelatihan & Evaluasi Model Klasifikasi (Membuat 6 Grafik)
py train_model_klasifikasi.py

# 5. Jalankan Aplikasi Web Dashboard Streamlit
py -m streamlit run app.py
```

---

## ⚠️ Keterbatasan Sistem & Catatan Akademik

1. **Metode Proxy Labeling:** Karena OLX tidak menyediakan data tanggal pasti kendaraan terjual, label *Cepat/Sedang/Lambat* dibentuk menggunakan simpangan harga jual terhadap estimasi harga wajar.
2. **Isolasi Data Leakage:** Dataset telah dipisah secara murni (80% Train & 20% Test) sebelum proses pelabelan dan penghitungan persentil tertile dilakukan untuk menjamin validitas pengujian.
3. **Peringatan Keyakinan Sampel:** Jika kombinasi kendaraan yang diinput penjual memiliki ketersediaan sampel sejenis $< 15$ unit di dataset, aplikasi Streamlit secara otomatis menampilkan peringatan keyakinan sampel rendah.