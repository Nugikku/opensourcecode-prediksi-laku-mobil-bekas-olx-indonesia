# 🚗 AutoValuate: Smart Liquidity & Pricing Decision System Mobil Bekas OLX Indonesia

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web_Dashboard-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Playwright](https://img.shields.io/badge/Playwright-Automated_Scraping-green.svg)](https://playwright.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**AutoValuate** adalah sistem *End-to-End Data Mining & Machine Learning* terintegrasi yang dirancang untuk menganalisis likuiditas penjualan dan memberikan rekomendasi keputusan harga mobil bekas di pasar daring Indonesia (studi kasus: marketplace **OLX Indonesia**). 

Sistem ini mencakup seluruh siklus *Knowledge Discovery in Databases (KDD)*: mulai dari **pengumpulan data dinamis dengan interceptor API**, **normalisasi teks slang & jargon otomotif lokal**, **rekayasa fitur pasar (market deviation & usage intensity)**, **evaluasi 5 algoritma klasifikasi (dengan 7 visualisasi analitik mendalam)**, hingga **Decision Support System (DSS) interaktif** dengan dua persona (*Penjual* vs *Pembeli*) yang tersedia dalam format **Web Dashboard Modern (Streamlit)** dan **Terminal CLI**.

---

## 📌 Daftar Isi
- [Latar Belakang & Masalah](#-latar-belakang--masalah)
- [Pembaruan Besar (Big Updates)](#-pembaruan-besar-big-updates)
- [Fitur Utama](#-fitur-utama)
- [Arsitektur & Alur Kerja Sistem](#-arsitektur--alur-kerja-sistem)
- [Struktur Direktori Proyek](#-struktur-direktori-proyek)
- [Tahapan & Metodologi](#-tahapan--metodologi)
  - [1. Pengumpulan Data (Scraping API Interception)](#1-pengumpulan-data-scrapingpy)
  - [2. Normalisasi Slang & NLP Otomotif](#2-normalisasi-slang--nlp-otomotif-kamus_slang_otomotifpy)
  - [3. Preprocessing & Ground Truth Likuiditas](#3-preprocessing--ground-truth-likuiditas-processingpy)
  - [4. Pelatihan Model & 7 Visualisasi Analitik](#4-pelatihan-model--7-visualisasi-analitik-train_model_klasifikasipy)
  - [5. Decision Support System Berbasis Web (Streamlit)](#5-decision-support-system-berbasis-web-apppy)
  - [6. Decision Support System Berbasis CLI](#6-decision-support-system-berbasis-cli-test_prediksi_klasifikasipy)
- [Hasil Komparasi & Evaluasi 5 Algoritma](#-hasil-komparasi--evaluasi-5-algoritma)
- [Panduan Instalasi & Penggunaan](#-panduan-instalasi--penggunaan)
- [Lisensi](#-lisensi)

---

## 💡 Latar Belakang & Masalah

Menetapkan dan menilai harga mobil bekas di pasar daring memiliki ketidakpastian tinggi:
1. **Bagi Penjual / Dealer**: Unit yang dibanderol terlalu tinggi (*overpriced*) akan mengendap lama (*slow-moving*), memicu depresiasi unit, dan menahan perputaran modal. Sebaliknya, harga terlalu rendah akan mengorbankan margin profit.
2. **Bagi Pembeli**: Sulit menilai apakah harga penawaran merupakan *fair deal* (sesuai kondisi pasar) atau kemahalan, serta minimnya referensi batas tawar realistis.

**AutoValuate** memetakan unit mobil ke dalam **3 Kategori Likuiditas Pasar**:
- **Cepat Laku (< 15 Hari)**: Unit sangat atraktif, harga di bawah rata-rata pasar, kilometer wajar/rendah, dan cepat terserap pembeli.
- **Laku Sedang (15 – 30 Hari)**: Unit berada pada titik ekuilibrium pasar standar dengan waktu penyerapan normal.
- **Lambat Laku (> 30 Hari / Rawan Macet)**: Unit terindikasi *overpriced* atau memiliki riwayat pemakaian tinggi sehingga rawan mengendap di marketplace.

---

## 🚀 Pembaruan Besar (Big Updates)

Proyek ini telah mengalami pembaruan komprehensif pada setiap lapis arsitekturnya:

1. 🌐 **Aplikasi Web Interaktif (Streamlit `app.py`)**:
   - Dashboard web modern dengan navigasi sidebar intuitif, filter merek/model bertingkat, dan kartu metrik analitik.
   - **Interactive Price Drop Simulator Slider**: Geser harga penawaran secara *real-time* untuk menguji bagaimana perubahan harga mengubah persentase deviasi pasar dan menggeser prediksi model ke kategori *Cepat Laku*.
   - Visualisasi distribusi probabilitas keyakinan model (*multi-class confidence bars*).
2. 🏷️ **Ground Truth Likuiditas Realistis (`processing.py`)**:
   - Integrasi indikasi sinyal terjual, batas durasi tayang mengambang, deviasi harga terhadap median segmen (ambang batas $+8\%$ dan $\le -3\%$), serta uji stres kilometer per tahun ($> 30.000\text{ KM/tahun}$).
   - Perluasan drastis kamus pengenal model mobil (mencakup puluhan seri baru: lini EV BYD/Chery/GWM/DFSK, lini diesel Isuzu/Ford/Chevrolet, dan segmen mobil mewah/Eropa).
3. 📈 **Suite 7 Visualisasi Analitik Resolusi Tinggi (`train_model_klasifikasi.py`)**:
   - Skrip pelatihan kini otomatis menghasilkan **7 visualisasi grafis 300 DPI** untuk evaluasi komprehensif data mining dan performa algoritma.
4. 🧠 **Peningkatan Akurasi & Benchmark Model**:
   - Pipeline klasifikasi mencapai **Akurasi 97.65% (Decision Tree)** dan **ROC-AUC 0.9911 (Gradient Boosting)** pada dataset bersih multi-kelas.

---

## ✨ Fitur Utama

- ⚡ **Playwright API Interceptor**: Scraping respons network JSON OLX secara asinkron tanpa terhambat beban rendering DOM HTML.
- 📚 **Kamus Slang Otomotif Indonesia**: Standardisasi kata singkatan dan slang pasar (`kamus_slang_otomotif.py`) mencakup transmisi, legalitas/pajak, riwayat servis, integritas insiden, dan istilah negosiasi.
- 📊 **Rekayasa Fitur Pasar (Domain-Driven)**:
  - `deviasi_harga_pasar`: Persentase selisih harga iklan terhadap median harga pasar grup `[model, tahun]`.
  - `km_per_tahun`: Intensitas rasio jarak tempuh riil terhadap usia mobil.
  - `tipe_penjual` & `ada_garansi`: Deteksi dealer vs individu dan sertifikasi inspeksi (*Otospector*, *OLXmobbi*, garansi bebas banjir).
- 🏆 **Multi-Model ML Benchmarking**: Evaluasi 5 model (Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting) berbasis pipeline `ColumnTransformer` (One-Hot Encoding & StandardScaler).
- 🎭 **Dual-Perspective Decision Support System (DSS)**:
  - **Persona Penjual**: Analisis daya serap pasar, probabilitas kelas, dan simulasi penyesuaian harga optimal.
  - **Persona Pembeli**: Evaluasi *Fair Deal / Best Price* vs *Overpriced* serta rekomendasi plafon harga tawar.
- 🎛️ **Dual Interface Support**: Jalankan melalui antarmuka visual web (**Streamlit**) atau konsol interaktif terminal (**CLI**).

---

## 🏗️ Arsitektur & Alur Kerja Sistem

```text
                  [ Marketplace OLX Indonesia ]
                                │
                                ▼ (Playwright Network Interception)
                         scraping.py
                                │
                                ▼
                     dataset_olx_mentah.csv
                                │
            ┌───────────────────┴───────────────────┐
            │                                       │
            ▼                                       ▼
  kamus_slang_otomotif.py                   DAFTAR_MODEL (Regex)
  (Normalisasi Slang & Teks)                (Ekstraksi Seri Mobil)
            │                                       │
            └───────────────────┬───────────────────┘
                                ▼
                          processing.py
       ┌─────────────────────────────────────────────────┐
       │ - Filter outlier harga & tahun                  │
       │ - Imputasi median odometer per tahun            │
       │ - Rekayasa fitur: deviasi pasar & KM/tahun      │
       │ - Labeling ground truth 3 kelas likuiditas      │
       └─────────────────────────────────────────────────┘
                                │
                                ▼
                     dataset_olx_bersih.csv
                                │
                                ▼
                   train_model_klasifikasi.py
       ┌─────────────────────────────────────────────────┐
       │ Pipeline: OneHotEncoder + StandardScaler        │
       │ Benchmark: LogReg, KNN, DT, RF, GradientBoost   │
       └─────────────────────────────────────────────────┘
                                │
             ┌──────────────────┴──────────────────┐
             ▼                                     ▼
  model_klasifikasi_penjualan.pkl        output_klasifikasi_<TANGGAL>/
  (Pipeline Model Terbaik)               ├── 7 Grafik Visualisasi 300 DPI
                                         └── ringkasan_metrik.csv
                                │
             ┌──────────────────┴──────────────────┐
             ▼                                     ▼
          app.py                      test_prediksi_klasifikasi.py
    (Streamlit Web GUI)                     (Terminal CLI)
  - Seleksi Persona                     - Navigasi 4 Kolom Rapi
  - Kartu Metrik Unit                   - Unit Selector Marketplace
  - Diagnosa Deal & Likuiditas          - Price Drop Simulator Iteratif
  - Slider Price Drop Simulator
```

---

## 📁 Struktur Direktori Proyek

```text
.
├── app.py                           # Aplikasi Web Dashboard Interaktif (Streamlit)
├── test_prediksi_klasifikasi.py     # Aplikasi Terminal Konsol Interaktif (CLI DSS)
├── train_model_klasifikasi.py       # Skrip pelatihan 5 model ML & generator 7 grafik evaluasi
├── processing.py                    # Pipeline data cleaning, regex extraction & feature engineering
├── kamus_slang_otomotif.py          # Kamus pemetaan normalisasi slang & jargon otomotif
├── scraping.py                      # Crawler otomatis berbasis Playwright network interception
├── dataset_olx_mentah.csv           # Hasil ekstraksi data mentah dari OLX
├── dataset_olx_bersih.csv           # Dataset bersih siap latih hasil preprocessing
├── model_klasifikasi_penjualan.pkl  # Serialized pipeline model machine learning terbaik
├── output_klasifikasi_23-09-2026/   # Artefak visualisasi & metrik evaluasi
│   ├── 1_komparasi_semua_metrik.png     # Grafik batang 5 metrik performa seluruh model
│   ├── 2_confusion_matrix_komparasi.png # 5 matriks konfusi berdampingan
│   ├── 3_kurva_roc_multiclass.png       # Kurva ROC One-vs-Rest model terbaik per kelas
│   ├── 4_feature_importance.png         # Peringkat 15 fitur paling berpengaruh
│   ├── 5_distribusi_target_kelas.png    # Distribusi proporsi kelas target (Ground Truth)
│   ├── 6_analisis_harga_vs_likuiditas.png # Boxplot deviasi harga vs kelas likuiditas
│   ├── 7_radar_chart_performa.png       # Radar chart profil keseimbangan metrik
│   └── ringkasan_metrik.csv             # Tabel rekapitulasi nilai evaluasi numerik
├── LICENSE                          # Lisensi proyek (MIT License)
└── README.md                        # Dokumentasi komprehensif proyek
```

---

## 🔬 Tahapan & Metodologi

### 1. Pengumpulan Data (`scraping.py`)
- Menggunakan Playwright Chromium async untuk membuka kategori mobil bekas di berbagai kota besar di Indonesia.
- Menyadap data langsung dari *network response JSON* OLX (`cat_id == 198`), mengabaikan postingan non-mobil (misal gadget/ponsel).
- Menghasilkan atribut mentah: `id_iklan`, `judul`, `deskripsi`, `merek`, `model`, `tahun`, `transmisi`, `jarak_tempuh`, `harga`, `tipe_penjual_badge`, `lokasi`, dan `tanggal_posting`.

### 2. Normalisasi Slang & NLP Otomotif (`kamus_slang_otomotif.py`)
Standardisasi kosakata pasar percakapan jual-beli di Indonesia:
- **Transmisi**: `mt`, `m/t`, `matic`, `at`, `tiptronic`, `cvt` $\rightarrow$ `manual` / `otomatis`.
- **Legalitas**: `pjk on/off`, `tgn1`, `anperorangan`, `kaleng`, `bpkb`, `stnk` $\rightarrow$ standarisasi teks.
- **Kondisi & Servis**: `low km`, `rec`, `bengkelresmi`, `kaki2`, `nyess`, `istmw`, `antik`.
- **Integritas**: `nobanjir`, `nolaka`, `otospector`, `olxmobbi`, `garansi`.
- **Skema Transaksi**: `bu`, `tdp`, `dp`, `angs`, `otr`, `nego halus`, `nett`.

### 3. Preprocessing & Ground Truth Likuiditas (`processing.py`)
- **Penyaringan Anomali**: Rentang harga valid Rp 25.000.000 s/d Rp 2.500.000.000; tahun perakitan 1995 s/d 2026.
- **Ekstraksi Seri/Model Lengkap**: Pencocokan regex terhadap puluhan model populer Jepang, Eropa, Korea, Amerika, hingga EV modern (BYD, Chery, Wuling EV, GWM, DFSK).
- **Imputasi Odometer**: Mengisi nilai kosong jarak tempuh berdasarkan median kelompok tahun perakitan.
- **Formulasi Fitur Pasar**:
  $$\text{Usia Mobil} = 2026 - \text{Tahun}$$
  $$\text{KM per Tahun} = \frac{\text{Jarak Tempuh}}{\max(1, \text{Usia Mobil})}$$
  $$\text{Deviasi Harga Pasar (\%)} = \frac{\text{Harga Iklan} - \text{Median Pasar}}{\text{Median Pasar}} \times 100$$
- **Logika Ground Truth Likuiditas Pasar (3 Kelas)**:
  1. Unit bertanda status `Terjual`: diklasifikasikan berdasarkan durasi tayang hingga laku ($<15$ hari = *Cepat*, $15\text{--}30$ hari = *Sedang*, $>30$ hari = *Lambat*).
  2. Unit yang belum terjual dan tayang $>30$ hari: diklasifikasikan sebagai *Lambat* (indikasi unit macet).
  3. Unit aktif baru ($1\text{--}30$ hari):
     - Jika $\text{Deviasi} > +8.0\%$ ATAU $\text{KM/Tahun} > 30.000\text{ KM} \rightarrow$ **Lambat** (*overpriced* / pemakaian berat).
     - Jika $\text{Deviasi} \le -3.0\%$ DAN $\text{KM/Tahun} \le 18.000\text{ KM} \rightarrow$ **Cepat** (harga murah & kondisi prima).
     - Selainnya $\rightarrow$ **Sedang** (rentang harga dan pemakaian wajar).

### 4. Pelatihan Model & 7 Visualisasi Analitik (`train_model_klasifikasi.py`)
Fitur input diproses melalui pipeline:
- **Kategorikal** (`merek`, `model`, `transmisi`, `tipe_penjual`, `ada_garansi`) $\rightarrow$ `OneHotEncoder(handle_unknown='ignore')`.
- **Numerikal** (`harga`, `tahun`, `jarak_tempuh`, `usia_mobil`, `km_per_tahun`, `deviasi_harga_pasar`) $\rightarrow$ `StandardScaler()`.

Skrip melatih 5 algoritma, memilih model dengan weighted F1-Score tertinggi, menyimpannya ke `model_klasifikasi_penjualan.pkl`, dan mengekspor 7 grafik visualisasi analitik ke folder `output_klasifikasi_<TANGGAL>/`:
1. `1_komparasi_semua_metrik.png`: Bar chart komparasi 5 metrik performa untuk 5 model.
2. `2_confusion_matrix_komparasi.png`: Matriks konfusi berdampingan untuk melihat distribusi prediksi vs ground truth.
3. `3_kurva_roc_multiclass.png`: Evaluasi kemampuan pemisahan kelas (One-vs-Rest) model terbaik.
4. `4_feature_importance.png`: Peringkat fitur paling berpengaruh dalam keputusan likuiditas.
5. `5_distribusi_target_kelas.png`: Proporsi sampel kelas *Cepat*, *Sedang*, dan *Lambat*.
6. `6_analisis_harga_vs_likuiditas.png`: Boxplot distribusi persentase deviasi harga terhadap kelas likuiditas.
7. `7_radar_chart_performa.png`: Radar chart keseimbangan metrik (Akurasi, Precision, Recall, F1, AUC).

### 5. Decision Support System Berbasis Web (`app.py`)
Aplikasi web modern berbasis **Streamlit**:
- Pemilihan Persona: *Penjual (Strategi Penjualan)* vs *Pembeli (Analisis Deal)*.
- Penelusuran katalog kendaraan: Merek $\rightarrow$ Model $\rightarrow$ Unit riil pasar.
- Dashboard metrik unit (Harga penawaran, Odometer, Durasi tayang, Median segmen, Deviasi pasar, Garansi, Tipe penjual).
- Indikator hasil prediksi model lengkap dengan bar progres probabilitas multi-kelas.
- **Interactive Price Drop Simulator Slider**: Geser harga penawaran secara bebas untuk melihat kalkulasi deviasi baru dan perubahan status prediksi likuiditas secara instan.

### 6. Decision Support System Berbasis CLI (`test_prediksi_klasifikasi.py`)
Aplikasi terminal interaktif dengan navigasi rapi:
- Menu seleksi merek dan model dalam 4 kolom sejajar.
- Penampil data riil marketplace dalam tabel terformat.
- Algoritma simulator penurunan harga bertahap (*price drop simulator*) otomatis.

---

## 📊 Hasil Komparasi & Evaluasi 5 Algoritma

Hasil pengujian pada data uji (*test set 20% stratified*) setelah rekayasa fitur ground truth likuiditas:

| Model Klasifikasi | Akurasi | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | ROC-AUC (OvR Weighted) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Decision Tree** | **97.65%** | **0.9768** | **0.9765** | **0.9760** | **0.9817** |
| 🥈 **Gradient Boosting** | 97.01% | 0.9698 | 0.9701 | 0.9698 | **0.9911** |
| 🥉 **Logistic Regression** | 81.84% | 0.8405 | 0.8184 | 0.8091 | 0.8906 |
| **Random Forest** | 81.84% | 0.8559 | 0.8184 | 0.8047 | 0.9773 |
| **K-Nearest Neighbors (KNN)** | 60.90% | 0.6055 | 0.6090 | 0.6071 | 0.7027 |

> **Analisis Model**: **Decision Tree** dan **Gradient Boosting** menunjukkan performa luar biasa dalam menangkap batasan keputusan (*decision boundary*) multi-fitur antara deviasi harga dan intensitas pemakaian kendaraan, dengan ROC-AUC mencapai $>0.98$.

---

## 🛠️ Panduan Instalasi & Penggunaan

### 1. Klon Repositori
```bash
git clone https://github.com/Nugikku/opensourcecode-prediksi-laku-mobil-bekas-olx-indonesia.git
cd opensourcecode-prediksi-laku-mobil-bekas-olx-indonesia
```

### 2. Buat & Aktifkan Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Pasang Dependensi
```bash
pip install pandas numpy scikit-learn matplotlib seaborn playwright streamlit
playwright install chromium
```

### 4. Menjalankan Pipeline & Aplikasi

#### A. Web Dashboard Interaktif (Rekomendasi Utama 🌟)
Jalankan aplikasi web berbasis Streamlit:
```bash
python -m streamlit run app.py
# atau
streamlit run app.py
```
*Aplikasi akan otomatis terbuka di browser pada alamat `http://localhost:8501`.*

#### B. Terminal CLI Interaktif
Jalankan pengujian via antarmuka konsol:
```bash
python test_prediksi_klasifikasi.py
```

#### C. Menjalankan Ulang Pipeline dari Awal (Opsional)
1. **Scraping Data OLX**:
   ```bash
   python scraping.py
   ```
2. **Preprocessing & Ground Truth Labeling**:
   ```bash
   python processing.py
   ```
3. **Training & Ekspor 7 Grafik Visualisasi**:
   ```bash
   python train_model_klasifikasi.py
   ```

---

## 💻 Cuplikan Antarmuka Sistem

### 1. Tampilan Web Dashboard (Streamlit `app.py`)
- **Header & Ringkasan Unit**:
  - `Harga Penawaran`: Rp 178.000.000 | `Odometer`: 2.500 KM | `Lama Tayang`: 1 Hari | `Status Riil`: Cepat
- **Diagnosa & Prediksi Model**:
  - `[✓] CEPAT LAKU (< 15 Hari)` | Probabilitas: `88.2% Cepat`, `11.8% Sedang`
  - Median Pasar: Rp 172.500.000 (Deviasi: $+3.2\%$) | Profil: Dealer | Garansi: Ya
- **Simulasi Penyesuaian Harga**:
  - Slider interaktif geser harga Rp 160.000.000 s/d Rp 210.000.000 dengan update prediksi real-time.

### 2. Tampilan Terminal CLI (`test_prediksi_klasifikasi.py`)
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
Tingkat Keyakinan : 88.20%
--------------------------------------------------------------------------------
KESIMPULAN & REKOMENDASI (SUDUT PANDANG PENJUAL):
• Penetapan harga iklan Anda sangat atraktif dan sesuai daya serap pasar.
• Unit diprediksi cepat diminati dan terjual dalam 2 minggu pertama.
================================================================================
```

---

## 📜 Lisensi

Proyek ini didistribusikan di bawah lisensi **MIT License**. Silakan lihat berkas [LICENSE](LICENSE) untuk informasi selengkapnya.