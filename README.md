# 🚗 AutoValuate: Prediksi Likuiditas & Sistem Keputusan Harga Mobil Bekas OLX Indonesia

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Playwright](https://img.shields.io/badge/Playwright-Automated_Scraping-green.svg)](https://playwright.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**AutoValuate** adalah sistem *End-to-End Data Mining & Machine Learning* yang dirancang untuk menganalisis likuiditas penjualan dan memberikan rekomendasi keputusan harga mobil bekas di pasar online Indonesia (studi kasus: marketplace **OLX Indonesia**). 

Sistem ini mengintegrasikan seluruh siklus penambangan data: mulai dari **pengumpulan data dinamis berbasis web interceptor**, **normalisasi teks bahasa pasar (slang otomotif)**, **rekayasa fitur pasar (market deviation & usage intensity)**, **benchmarking 5 model klasifikasi**, hingga **Decision Support System (DSS) interaktif** dengan dua persona (*Penjual* vs *Pembeli*) yang dilengkapi fitur **Price Drop Simulator**.

---

## 📌 Daftar Isi
- [Latar Belakang & Masalah](#-latar-belakang--masalah)
- [Fitur Utama](#-fitur-utama)
- [Arsitektur & Alur Kerja Sistem](#-arsitektur--alur-kerja-sistem)
- [Struktur Direktori](#-struktur-direktori)
- [Tahapan & Metodologi](#-tahapan--metodologi)
  - [1. Data Collection (Scraping)](#1-data-collection-scrapingpy)
  - [2. Normalisasi Slang & NLP Otomotif](#2-normalisasi-slang--nlp-otomotif-kamus_slang_otomotifpy)
  - [3. Preprocessing & Feature Engineering](#3-preprocessing--feature-engineering-processingpy)
  - [4. Pemodelan & Evaluasi](#4-pemodelan--evaluasi-train_model_klasifikasipy)
  - [5. Sistem Keputusan Interaktif (DSS)](#5-sistem-keputusan-interaktif-test_prediksi_klasifikasipy)
- [Hasil Komparasi Model](#-hasil-komparasi-model)
- [Panduan Instalasi & Penggunaan](#-panduan-instalasi--penggunaan)
- [Lisensi](#-lisensi)

---

## 💡 Latar Belakang & Masalah

Menetapkan harga mobil bekas di pasar daring merupakan tantangan tersendiri bagi penjual maupun pembeli:
1. **Bagi Penjual / Dealer**: Penetapan harga yang terlalu tinggi (*overpriced*) menyebabkan unit mengendap lama (*slow-moving*), memicu biaya modal tertahan dan depresiasi nilai. Sebaliknya, penetapan harga terlalu murah dapat merugikan potensi keuntungan.
2. **Bagi Pembeli**: Sulit membedakan unit yang merupakan *fair deal* (harga bersaing & kondisi wajar) dengan unit yang kemahalan, serta minimnya acuan batas bawah/atas saat negosiasi.

**AutoValuate** mengatasi masalah ini dengan memetakan unit mobil ke dalam **3 Kategori Likuiditas**:
- **Cepat Laku (< 15 Hari)**: Unit memiliki daya tarik pasar tinggi, rasio harga kompetitif, dan probabilitas perputaran cepat.
- **Laku Sedang (15 – 30 Hari)**: Unit berada pada harga pasar wajar dengan tempo penyerapan standar.
- **Lambat Laku (> 30 Hari / Rawan Macet)**: Unit terindikasi *overpriced* atau kurang kompetitif sehingga membutuhkan penyesuaian harga.

---

## ✨ Fitur Utama

- 🌐 **Dynamic Network Interception**: Scraping otomatis menggunakan Playwright yang menangkap langsung *payload* JSON API OLX tanpa terbebani rendering DOM browser yang lambat.
- 📖 **Domain-Specific Slang Normalizer**: Kamus khusus slang otomotif Indonesia (`kamus_slang_otomotif.py`) untuk standardisasi istilah transaksi, transmisi, surat-surat, hingga kondisi fisik mobil.
- ⚙️ **Advanced Market Feature Engineering**:
  - `deviasi_harga_pasar`: Deviasi persentase harga iklan terhadap median pasar berdasarkan kombinasi `[model, tahun]`.
  - `km_per_tahun`: Intensitas pemakaian riil kendaraan per tahun usia mobil.
  - `tipe_penjual` & `ada_garansi`: Deteksi profil dealer/showroom dan inspeksi sertifikasi (misal: *Otospector*, *OLXmobbi*).
- 🏆 **Multi-Model Machine Learning Pipeline**: Pelatihan dan komparasi 5 algoritma klasifikasi dengan *ColumnTransformer* (One-Hot Encoding & Standard Scaling).
- 🧭 **Dual-Perspective Decision Support System (DSS)**:
  - **Persona Penjual**: Analisis likuiditas iklan, probabilitas keyakinan model, dan **Price Drop Simulator** (simulasi penurunan bertahap harga iklan untuk mengetahui titik temu agar unit berpindah ke kategori *Cepat Laku*).
  - **Persona Pembeli**: Evaluasi kelayakan penawaran (*Fair Deal* vs *Overpriced*) dan kalkulasi plafon harga tawar realistis.
- 🖥️ **Interactive Multi-Column Terminal UI**: Navigasi pemilihan merek, model, dan unit riil berbasis nomor dengan format kolom rapi.

---

## 🏗️ Arsitektur & Alur Kerja Sistem

```text
  [ OLX Indonesia Web API ]
              │
              ▼ (Playwright Network Interception)
       scraping.py
              │
              ▼
   dataset_olx_mentah.csv
              │
              ├──────► kamus_slang_otomotif.py (Standardisasi Slang)
              ▼
       processing.py (Regex Model, Feature Engineering: Deviasi, KM/Thn, Label Likuiditas)
              │
              ▼
   dataset_olx_bersih.csv
              │
              ▼
  train_model_klasifikasi.py
  (ColumnTransformer Pipeline: Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting)
              │
              ├──► model_klasifikasi_penjualan.pkl (Model Terbaik)
              └──► output_klasifikasi_*/ (Confusion Matrix, Komparasi Metrik)
              │
              ▼
  test_prediksi_klasifikasi.py (AutoValuate CLI)
  ├── Sudut Pandang Penjual (Analisis Likuiditas + Price Drop Simulator)
  └── Sudut Pandang Pembeli (Fair Deal Evaluator + Strategi Nego)
```

---

## 📁 Struktur Direktori

```text
.
├── dataset_olx_mentah.csv           # Hasil ekstraksi mentah dari scraper OLX
├── dataset_olx_bersih.csv           # Dataset hasil preprocessing dan feature engineering
├── kamus_slang_otomotif.py          # Kamus kamus pemetaan singkatan & slang otomotif
├── model_klasifikasi_penjualan.pkl  # Serialized pipeline model ML terbaik
├── processing.py                    # Pipeline pembersihan data, parsing & rekayasa fitur
├── scraping.py                      # Skrip web scraping berbasis Playwright async
├── train_model_klasifikasi.py       # Pelatihan, benchmark 5 algoritma, dan evaluasi
├── test_prediksi_klasifikasi.py     # Aplikasi CLI interaktif Sistem Pendukung Keputusan (DSS)
├── output_klasifikasi_23-09-2026/   # Folder artefak evaluasi model
│   ├── confusion_matrix.png         # Visualisasi matriks konfusi model terbaik
│   ├── komparasi_metrik.png         # Grafik batang komparasi Akurasi & F1-Score
│   └── ringkasan_metrik.csv         # Tabel rekapitulasi nilai evaluasi seluruh model
├── LICENSE                          # Lisensi proyek (MIT License)
└── README.md                        # Dokumentasi proyek
```

---

## 🔬 Tahapan & Metodologi

### 1. Data Collection (`scraping.py`)
- Menggunakan Playwright Chromium async untuk membuka halaman kategori mobil bekas di berbagai wilayah Indonesia (Jabodetabek, Jawa Barat, Jawa Timur, Bali, dsb.).
- Memanfaatkan pendengar event `page.on("response", ...)` untuk mencegat respon data berformat JSON dari API OLX.
- Menyaring data noise (mengecualikan kategori non-mobil seperti gadget/elektronik yang salah kamar).
- Menyimpan atribut mentah: `id_iklan`, `judul`, `deskripsi`, `merek`, `model`, `tahun`, `transmisi`, `jarak_tempuh`, `harga`, `tipe_penjual_badge`, `lokasi`, dan `tanggal_posting`.

### 2. Normalisasi Slang & NLP Otomotif (`kamus_slang_otomotif.py`)
Menerapkan standardisasi kata terhadap teks bahasa percakapan sehari-hari pada judul dan deskripsi iklan yang mencakup 6 domain:
1. **Transmisi & Mesin**: `mt`/`m/t` $\rightarrow$ *manual*, `matic`/`at` $\rightarrow$ *otomatis*, `kering` $\rightarrow$ *mesin kering*.
2. **Legalitas & Dokumen**: `pjk on/off` $\rightarrow$ *pajak hidup/mati*, `tgn1` $\rightarrow$ *tangan pertama*, `anperorangan` $\rightarrow$ *milik pribadi*.
3. **Kondisi & Odometer**: `low km` $\rightarrow$ *kilometer rendah*, `rec` $\rightarrow$ *catatan servis*, `kaki2` $\rightarrow$ *suspensi*, `nyess` $\rightarrow$ *dingin*.
4. **Integritas Unit**: `nobanjir` $\rightarrow$ *bebas banjir*, `nolaka` $\rightarrow$ *bebas tabrak*, `otospector` $\rightarrow$ *inspeksi otospector*.
5. **Skema Transaksi**: `bu` $\rightarrow$ *butuh uang*, `tdp` $\rightarrow$ *total uang muka*, `angs` $\rightarrow$ *angsuran*, `otr` $\rightarrow$ *harga jalan*.
6. **Slang Harian**: `yg`, `dgn`, `udh`, `blm`, `gk`, dll.

### 3. Preprocessing & Feature Engineering (`processing.py`)
- **Pembersihan Outlier**: Filter rentang harga realistis (Rp 25.000.000 s/d Rp 2.500.000.000) dan tahun pembuatan (1995 – 2026).
- **Regex Model Extractor**: Ekstraksi nama model yang akurat dari judul iklan berdasarkan pustaka model populer (Toyota, Honda, Mazda, Mitsubishi, Suzuki, Hyundai, Wuling, EV baru BYD/Chery, hingga lini Eropa).
- **Penanganan Nilai Kosong**: Imputasi jarak tempuh menggunakan nilai median berdasarkan tahun perakitan kendaraan.
- **Labeling Target Likuiditas (3 Kelas)**:
  - *Cepat*: Iklan terindikasi `Terjual` ATAU durasi tayang $< 15$ hari.
  - *Sedang*: Durasi tayang antara $15$ sampai $30$ hari.
  - *Lambat*: Durasi tayang $> 30$ hari.
- **Fitur Finansial & Teknis**:
  - $\text{Usia Mobil} = 2026 - \text{Tahun}$
  - $\text{KM per Tahun} = \frac{\text{Jarak Tempuh}}{\max(1, \text{Usia Mobil})}$
  - $\text{Deviasi Harga Pasar (\%)} = \frac{\text{Harga Iklan} - \text{Median Pasar}}{\text{Median Pasar}} \times 100$

### 4. Pemodelan & Evaluasi (`train_model_klasifikasi.py`)
Fitur yang digunakan untuk pelatihan:
- **Kategorikal**: `['merek', 'model', 'transmisi', 'tipe_penjual', 'ada_garansi']` diolah dengan `OneHotEncoder(handle_unknown='ignore')`.
- **Numerikal**: `['harga', 'tahun', 'jarak_tempuh', 'usia_mobil', 'km_per_tahun', 'deviasi_harga_pasar']` distandarisasi menggunakan `StandardScaler()`.

Pengujian dilakukan menggunakan pembagian data 80:20 (*stratified train-test split*) terhadap 5 algoritma:
1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree Classifier
4. Random Forest Classifier
5. Gradient Boosting Classifier

Pipeline model dengan F1-Score terbaik secara otomatis diserialisasi dan disimpan ke dalam file `model_klasifikasi_penjualan.pkl`.

### 5. Sistem Keputusan Interaktif (`test_prediksi_klasifikasi.py`)
Aplikasi konsol yang memberikan pengalaman interaktif bagi pengguna:
1. Memilih sudut pandang (Penjual vs Pembeli).
2. Memilih Merek dan Model dari daftar berkolom rapi.
3. Meninjau unit-unit riil yang ditemukan di dataset.
4. Mendapatkan kartu evaluasi analitik:
   - Status tayang riil vs Prediksi likuiditas model + tingkat keyakinan (*confidence level*).
   - Selisih deviasi harga terhadap median segmen lokal.
   - **Price Drop Simulator**: Rekomendasi harga baru bagi penjual atau batas atas penawaran bagi pembeli.

---

## 📊 Hasil Komparasi Model

Evaluasi model pada data uji (*test set*) 3 kelas (*Cepat*, *Sedang*, *Lambat*):

| Model | Akurasi | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | ROC-AUC (OvR Weighted) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** 🏆 | **84.37%** | **0.8382** | **0.8437** | **0.7742** | **0.6632** |
| **Logistic Regression** | 84.37% | 0.8382 | 0.8437 | 0.7742 | 0.6287 |
| **Gradient Boosting** | 83.30% | 0.7417 | 0.8330 | 0.7721 | 0.6474 |
| **KNN** | 81.58% | 0.7167 | 0.8158 | 0.7604 | 0.6209 |
| **Decision Tree** | 82.01% | 0.7053 | 0.8201 | 0.7584 | 0.6158 |

> **Catatan**: Random Forest dipilih sebagai model utama karena memiliki nilai **ROC-AUC tertinggi (0.6632)** dan ketahanan terhadap *overfitting* pada fitur tabular berkategori jamak.

---

## 🚀 Panduan Instalasi & Penggunaan

### 1. Klon Repositori
```bash
git clone https://github.com/Nugikku/opensourcecode-prediksi-laku-mobil-bekas-olx-indonesia.git
cd opensourcecode-prediksi-laku-mobil-bekas-olx-indonesia
```

### 2. Buat & Aktifkan Virtual Environment (Opsional tapi Direkomendasikan)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Pasang Dependensi
Pastikan library yang dibutuhkan terinstal di sistem Anda:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn playwright
playwright install chromium
```

### 4. Menjalankan Pipeline Lengkap

#### A. Scraping Data OLX (Opsional jika ingin memperbarui dataset mentah)
```bash
python scraping.py
```
*Output: `dataset_olx_mentah.csv`*

#### B. Preprocessing & Feature Engineering
```bash
python processing.py
```
*Output: `dataset_olx_bersih.csv`*

#### C. Latih Model & Benchmark
```bash
python train_model_klasifikasi.py
```
*Output: `model_klasifikasi_penjualan.pkl` dan visualisasi metrik di folder `output_klasifikasi_<TANGGAL>/`*

#### D. Menjalankan Sistem Keputusan Interaktif (DSS)
```bash
python test_prediksi_klasifikasi.py
```

---

## 🖥️ Contoh Tampilan Penggunaan (CLI)

```text
================================================================================
     AUTOVALUATE: SMART LIQUIDITY & PRICING DECISION SYSTEM (3 KELAS)    
================================================================================
Pilih Sudut Pandang Pengguna:
1. Penjual (Evaluasi kecepatan laku & simulasi optimasi harga)
2. Pembeli (Evaluasi kelayakan deal & strategi tawar-menawar)
Pilih (1/2): 1

================================================================================
Pilih Merek Mobil:
[1 ] Bmw              [2 ] Byd              [3 ] Chery            [4 ] Daihatsu     
[5 ] Ford             [6 ] Gwm              [7 ] Honda            [8 ] Hyundai      
[9 ] Kia              [10] Lexus            [11] Mazda            [12] Mercedes-Benz
[13] Mini             [14] Mini Cooper      [15] Mitsubishi       [16] Nissan       
[17] Subaru           [18] Suzuki           [19] Toyota           [20] Wuling       

Pilih nomor merek (1-20): 7
...
================================================================================
                   HASIL ANALISIS SISTEM KEPUTUSAN RIIL
================================================================================
Judul Iklan Riil  : Honda Brio E CVT 2024
Spesifikasi       : Honda Brio (2024) | Transmisi: Otomatis
Odometer          : 2,500 KM (Rata-rata: 1,250 KM/Tahun)
Profil Penjual    : Dealer | Status Garansi: Ya
--------------------------------------------------------------------------------
RIWAYAT TAYANG & DATA PASAR:
- Tanggal Posting : 2026-09-23T16:41:35+07:00
- Durasi Tayang   : 1 Hari di Marketplace
- Status Asli     : Kategori 'Cepat'
- Harga Iklan     : Rp 178,000,000
- Median Pasar    : Rp 172,500,000 (Deviasi: +3.2%)
--------------------------------------------------------------------------------
PREDIKSI MODEL    : [✓] CEPAT LAKU (< 15 Hari)
Tingkat Keyakinan : 86.85%
--------------------------------------------------------------------------------
KESIMPULAN & REKOMENDASI (SUDUT PANDANG PENJUAL):
• Penetapan harga iklan Anda sangat atraktif dan sesuai daya serap pasar.
• Unit diprediksi cepat diminati dan terjual dalam 2 minggu pertama.
================================================================================
```

---

## 📜 Lisensi

Proyek ini didistribusikan di bawah lisensi **MIT License**. Silakan lihat file [LICENSE](LICENSE) untuk informasi lebih detail.