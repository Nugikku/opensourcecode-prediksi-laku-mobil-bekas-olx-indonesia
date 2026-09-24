# 🚗 AutoLiquid DSS: Automotive Valuation & Market Liquidity Decision Support System

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/scikit--learn-Regression-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Playwright](https://img.shields.io/badge/Playwright-Automated_Scraping-green.svg)](https://playwright.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**AutoLiquid DSS** adalah Sistem Pendukung Keputusan (*Decision Support System* / DSS) otomotif berbasis AI cerdas yang mengintegrasikan model **Machine Learning Regresi** dengan **Rule-Based Liquidity & Risk Engine**. 

Sistem ini dirancang khusus untuk memecahkan tantangan ketiadaan data historis transaksi publik (*ground truth sold items*) di pasar mobil bekas daring dengan memprediksi **Nilai Wajar Pasar (*Fair Market Value*)** serta memproyeksikan **Tingkat Likuiditas Penjualan (Cepat, Sedang, Lambat Laku)**.

Sistem mendukung arsitektur **Dual-Persona**:
1. **Persona Penjual:** Evaluasi mandiri terhadap spesifikasi kendaraan, rencana harga pasang, dan pemindaian teks deskripsi iklan secara *real-time* berbasis Natural Language Processing (NLP) untuk memastikan unit berdaya serap tinggi tanpa mengorbankan margin.
2. **Persona Pembeli:** Analisis kewajaran harga terhadap unit aktif marketplace untuk mendeteksi *Fair Deal*, unit kemahalan (*overpriced*), serta sistem proteksi dini terhadap indikasi jebakan skema kredit/DP tipuan (*Total Down Payment / TDP Scam*).

---

## 📌 Daftar Isi
- [Latar Belakang & Platform Constraint](#-latar-belakang--platform-constraint)
- [Fitur Utama Sistem](#-fitur-utama-sistem)
- [Logika Sistem & Arsitektur DSS](#-logika-sistem--arsitektur-dss)
  - [1. Algoritma Valuasi (Machine Learning Core)](#1-algoritma-valuasi-machine-learning-core)
  - [2. Floor Price Safety Clamp (Pengaman Ekstrapolasi)](#2-floor-price-safety-clamp-pengaman-ekstrapolasi)
  - [3. Logika Penentuan Likuiditas (Decision Engine)](#3-logika-penentuan-likuiditas-decision-engine)
  - [4. Deteksi Risiko Anomali (TDP & Credit Scam Warning)](#4-deteksi-risiko-anomali-tdp--credit-scam-warning)
  - [5. Interactive Price Drop Simulator](#5-interactive-price-drop-simulator)
- [Atribut Sistem & Rekayasa Fitur](#-atribut-sistem--rekayasa-fitur)
- [Struktur Repositori](#-struktur-repositori)
- [Hasil Komparasi & Evaluasi 5 Model Regresi](#-hasil-komparasi--evaluasi-5-model-regresi)
- [Panduan Instalasi & Penggunaan](#-panduan-instalasi--penggunaan)
  - [1. Kloning Repositori & Virtual Environment](#1-kloning-repositori--persiapan-environment)
  - [2. Instalasi Dependensi](#2-instalasi-dependensi)
  - [3. Menjalankan Dashboard Web (Streamlit)](#3-menjalankan-dashboard-web-streamlit)
  - [4. Menjalankan Pengujian Terminal (CLI)](#4-menjalankan-pengujian-terminal-cli)
  - [5. Eksekusi Pipeline Data Mining dari Awal](#5-eksekusi-pipeline-data-mining-dari-awal)
- [Lisensi](#-lisensi)

---

## 📌 Latar Belakang & Platform Constraint

Pada marketplace otomotif daring seperti **OLX Indonesia**, iklan kendaraan yang telah berstatus **"Terjual"** secara otomatis dicabut dari etalase pencarian publik (*unlisted*) demi menjaga kebersihan katalog etalase aktif. Akibatnya, peneliti maupun pelaku pasar tidak memiliki akses ke data historis transaksi publik (*ground truth sold transaction data*). 

Ketiadaan data historis penjualan langsung ini menyebabkan pendekatan klasifikasi langsung rentan terhadap bias data baru (*sampling bias*). **AutoLiquid DSS** menyelesaikan batasan struktural platform ini melalui metodologi **Proxy & Deviational Benchmark Model**:
- **Baseline Pembelajaran Pasar:** Memanfaatkan ribuan listing aktif di marketplace sebagai representasi penawaran pasar riil untuk melatih model regresi nilai wajar.
- **Determinan Daya Serap (*Market Absorption*):** Menghitung selisih deviasi persentase antara harga penawaran iklan terhadap estimasi nilai wajar pasar sebagai indikator utama kecepatan unit terserap oleh calon pembeli.

---

## ✨ Fitur Utama Sistem

- 🧠 **Hybrid DSS Architecture**: Sinergi antara model prediktif machine learning (regresi kontinu) dan aturan heuristik domain pasar (*expert rule-based engine*).
- 🔍 **Real-Time NLP Text Scanner**: Pemindaian teks leksikal otomatis terhadap deskripsi iklan untuk mendeteksi garansi, riwayat servis, bebas insiden banjir/tabrak, serta indikasi skema kredit/TDP.
- 🛡️ **TDP Scam Early Warning System**: Proteksi otomatis bagi calon pembeli dari penjual yang memasang harga uang muka (DP) pada kolom harga tunai total kendaraan.
- 🎚️ **Interactive Price Drop Simulator**: Fitur simulasi harga interaktif (slider dinamis) untuk menguji skenario penurunan harga dan mengetahui titik temu nominal agar unit berpindah status menjadi **Cepat Laku**.
- 🔒 **Floor Price Safety Clamp**: Algoritma penjaga batas bawah harga agar mobil berumur di atas 20 tahun tidak mengalami anomali harga bernilai negatif atau tidak masuk akal.
- 💻 **Dual-Interface Implementation**: Tersedia dalam antarmuka web modern berbasis **Streamlit (`app.py`)** dan konsol terminal interaktif **CLI (`test_prediksi_klasifikasi.py`)**.

---

## 🧠 Logika Sistem & Arsitektur DSS

Sistem bekerja melalui alur bertingkat (*multi-stage pipeline*):

```text
                    [ INPUT DATA ]
           ┌───────────────┴───────────────┐
   (Persona Penjual)               (Persona Pembeli)
Form Input Mandiri + Teks        Pilih Iklan Marketplace
           └───────────────┬───────────────┘
                           │
        [ NLP Real-Time Scanning & Extraction ]
      - Deteksi Jaminan Garansi (Bebas Banjir, Servis)
      - Deteksi Indikasi Jebakan Kredit / TDP
                           │
           [ Machine Learning Core: Regresi ]
      Estimasi Nilai Wajar Pasar (Fair Market Value)
      + Floor Price Safety Clamp (Minimal Rp 20 Jt)
                           │
           [ Financial & Odometer Deviation ]
     Deviasi (%) = ((Harga Iklan - Harga Wajar) / Harga Wajar) * 100
     Rasio Beban = Total KM / Usia Kendaraan (KM/Tahun)
                           │
         [ Rule-Based Liquidity & Risk Engine ]
           ┌───────────────┼───────────────┐
    Deviasi <= -3%   -3% < Dev <= +7%    Deviasi > +7%
          │                │               │
    [ CEPAT LAKU ]   [ LAKU SEDANG ] [ LAMBAT LAKU ]
    (< 15 Hari)      (15 - 30 Hari)  (> 30 Hari)
                           │
    [ Rekomendasi Dinamis & Price Drop Simulator ]
```

---

### 1. Algoritma Valuasi (Machine Learning Core)
Inti penilaian estimasi harga menggunakan algoritma regresi yang dilatih dan dikomparasikan performanya menggunakan pipeline `ColumnTransformer` (OneHotEncoder untuk fitur kategorik & StandardScaler untuk fitur numerik):
- **Linear Regression & Ridge Regression** (Model Parametrik Baseline)
- **Decision Tree Regressor**
- **Random Forest Regressor** (Ensemble Bagging)
- **Gradient Boosting Regressor** (Ensemble Boosting)

Pipeline model terbaik secara otomatis diserialisasi dan disimpan ke dalam file `model_regresi_harga.pkl`.

### 2. Floor Price Safety Clamp (Pengaman Ekstrapolasi)
Untuk unit kendaraan dengan usia di atas 20 tahun (misalnya mobil hobi/kolektor tahun 1990-an), kurva linier berpotensi memproyeksikan harga negatif atau terlampau jatuh akibat depresiasi matematis. Sistem menerapkan fungsi pengaman batas bawah:

$$\text{Harga Wajar Final} = \max\Big(\text{Prediksi ML},\; 0.55 \times \text{Median Pasar Sejenis},\; \text{Rp } 20.000.000\Big)$$

### 3. Logika Penentuan Likuiditas (Decision Engine)
Status penyerapan unit ditentukan melalui aturan heuristik berbasis kondisi pasar:
- **Cepat Laku ($< 15$ Hari):** Deviasi harga penawaran $\le -3.0\%$. 
  > *Pengecualian*: Bila intensitas pemakaian kendaraan tinggi ($> 30.000\text{ KM/tahun}$) tanpa disertai jaminan garansi/servis resmi, status diturunkan ke kelas **Sedang**.
- **Laku Sedang ($15 - 30$ Hari):** Deviasi harga berada pada rentang pasar seimbang ($-3.0\% < \text{Deviasi} \le +7.0\%$).
- **Lambat Laku ($> 30$ Hari / Tertahan):** Deviasi harga penawaran $> +7.0\%$ (*overpriced* terhadap pasar wajar).

### 4. Deteksi Risiko Anomali (TDP & Credit Scam Warning)
Jika deviasi harga penawaran bernilai $\le -35.0\%$ dan teks deskripsi memuat pola leksikal uang muka/angsuran (`tdp`, `dp minim`, `angsuran`, `cicilan`, `paket kredit`), sistem secara otomatis menyalakan alarm:
> ⚠️ **WASPADA INDIKASI HARGA JEBAKAN (TDP/KREDIT)**: Harga penawaran terindikasi sebagai skema uang muka (DP), bukan pelunasan tunai keseluruhan unit mobil.

### 5. Interactive Price Drop Simulator
Fitur simulator yang memungkinkan penjual atau pembeli menggeser slider harga (rentang $-40\%$ hingga $+40\%$ dari harga wajar) untuk melihat respons pasar seketika:
- Rekalkulasi persentase deviasi pasar baru secara instan.
- Proyeksi perpindahan zona likuiditas (*Cepat*, *Sedang*, *Lambat*).
- Rekomendasi nominal target realistis untuk strategi transaksi cepat (*fast sale*) atau penawaran harga wajar (*fair offer*).

---

## 📊 Atribut Sistem & Rekayasa Fitur

Sistem beroperasi menggunakan 9 atribut inti:

| No | Atribut | Tipe Data | Sumber / Metode Ekstraksi | Fungsi dalam Sistem |
| :---: | :--- | :---: | :--- | :--- |
| **1** | `merek` | Kategorik | Metadata Scraping / Input | Identitas pabrikan kendaraan |
| **2** | `model` | Kategorik | Ekstraksi Regex Judul / Input | Seri spesifik mobil (Toyota Avanza, Honda Brio, dll.) |
| **3** | `tahun` | Numerik | Ekstraksi Regex / Input | Tahun perakitan (penentu kurva depresiasi) |
| **4** | `transmisi` | Kategorik | Ekstraksi Teks / Input | Tipe transmisi (`Automatic` vs `Manual`) |
| **5** | `jarak_tempuh` | Numerik | Pembersihan Odometer / Input | Total kilometer pemakaian kendaraan |
| **6** | `usia_mobil` | Numerik | Feature Engineering ($2026 - \text{Tahun}$) | Usia riil unit kendaraan |
| **7** | `km_per_tahun` | Numerik | Feature Engineering ($\text{KM} / \text{Usia}$) | Indikator intensitas pemakaian kendaraan |
| **8** | `tipe_penjual` | Kategorik | Metadata Akun / NLP Deskripsi | Profil penjual (`Individu` vs `Dealer`) |
| **9** | `ada_garansi` | Biner | NLP Regex Deskripsi | Poin nilai tambah proteksi (`Ya` vs `Tidak`) |
| **10** | `indikasi_kredit_tdp` | Biner | NLP Regex Deskripsi | Peringatan risiko skema kredit/DP |

---

## 📁 Struktur Repositori

```text
.
├── app.py                           # Antarmuka Dashboard DSS Interaktif (Streamlit)
├── test_prediksi_klasifikasi.py     # Skrip inferensi & DSS berbasis konsol Terminal (CLI)
├── train_model_klasifikasi.py       # Pelatihan 5 model regresi & visualisasi metrik evaluasi
├── processing.py                    # Pipeline preprocessing data, ekstraksi regex & NLP
├── kamus_slang_otomotif.py          # Modul kamus normalisasi istilah dan singkatan otomotif
├── scraping.py                      # Skrip web scraping berbasis Playwright async
├── dataset_olx_mentah.csv           # Dataset awal hasil scraping marketplace
├── dataset_olx_bersih.csv           # Dataset bersih hasil normalisasi & feature engineering
├── model_regresi_harga.pkl          # Serialisasi model regresi terbaik (Pipeline Scikit-Learn)
├── output_regresi_24-09-2026/       # Folder visualisasi evaluasi model & metrik
│   ├── 1_komparasi_metrik_regresi.png   # Grafik batang komparasi metrik 5 model regresi
│   ├── 2_actual_vs_predicted.png        # Scatter plot harga aktual vs harga prediksi
│   ├── 3_distribusi_residual.png        # Kurva histogram distribusi residual error
│   └── ringkasan_metrik_regresi.csv     # Rekapitulasi nilai metrik numerik seluruh model
├── LICENSE                          # Lisensi proyek (MIT License)
└── README.md                        # Dokumentasi komprehensif teknis proyek
```

---

## 📈 Hasil Komparasi & Evaluasi 5 Model Regresi

Seluruh model dievaluasi pada data uji (*test set 20%*) menggunakan 4 metrik standar:
- **$R^2$ Score**: Proporsi variansi harga pasar yang dapat dijelaskan oleh model (mendekati 1.0 semakin baik).
- **MAE (*Mean Absolute Error*)**: Rata-rata selisih mutlak nominal prediksi terhadap harga aktual (satuan Rupiah).
- **RMSE (*Root Mean Squared Error*)**: Sensitivitas error terhadap deviasi nilai ekstrem / *outlier* (satuan Rupiah).
- **MAPE (*Mean Absolute Percentage Error*)**: Rata-rata error relatif dalam satuan persentase.

### Tabel Rekapitulasi Evaluasi Model Regresi

| Model Regresi | $R^2$ Score | MAE (Rp) | RMSE (Rp) | MAPE (%) | Status Model |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Ridge Regression** | **0.8172** | 102.994.325 | **162.418.917** | 41.16% | **Model Terbaik Terpilih ($R^2$ & RMSE)** |
| 🥈 **Linear Regression** | 0.8153 | 101.207.619 | 163.240.791 | 41.47% | Model Parametrik Baseline |
| 🥉 **Gradient Boosting Regressor** | 0.7684 | **84.235.380** | 182.785.340 | **25.35%** | **MAE & MAPE Terendah** |
| **Random Forest Regressor** | 0.7103 | 103.215.480 | 204.431.746 | 35.41% | Model Ensemble Non-Linier |
| **Decision Tree Regressor** | 0.5172 | 133.740.693 | 263.934.950 | 49.34% | Model Pohon Tunggal |

> **Analisis Model**: **Ridge Regression** terpilih sebagai *backbone* utama valuasi harga karena memiliki skor **$R^2$ tertinggi (0.8172)** dan **RMSE terendah**, serta stabil terhadap multikolinearitas fitur kategorik *One-Hot Encoding*. Sementara **Gradient Boosting** unggul dalam menghasilkan **MAPE terendah (25.35%)**.

---

## 🚀 Panduan Instalasi & Penggunaan

### 1. Kloning Repositori & Persiapan Environment
```bash
git clone https://github.com/Nugikku/opensourcecode-prediksi-laku-mobil-bekas-olx-indonesia.git
cd opensourcecode-prediksi-laku-mobil-bekas-olx-indonesia
```

Buat dan aktifkan virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 2. Instalasi Dependensi
```bash
pip install pandas numpy scikit-learn matplotlib streamlit playwright
playwright install chromium
```

### 3. Menjalankan Dashboard Web (Streamlit)
Rekomendasi antarmuka visual utama:
```bash
python -m streamlit run app.py
# atau
streamlit run app.py
```
*Aplikasi akan otomatis terbuka pada peramban web di alamat `http://localhost:8501`.*

### 4. Menjalankan Pengujian Terminal (CLI)
Jika ingin menggunakan inferensi melalui antarmuka konsol/terminal:
```bash
python test_prediksi_klasifikasi.py
```

### 5. Eksekusi Pipeline Data Mining dari Awal (Opsional)
Jika Anda ingin mengambil ulang dataset atau melatih ulang model:
1. **Scraping Data OLX**:
   ```bash
   python scraping.py
   ```
2. **Pembersihan Data, Regex Parsing & NLP**:
   ```bash
   python processing.py
   ```
3. **Pelatihan Model & Ekspor Visualisasi**:
   ```bash
   python train_model_klasifikasi.py
   ```

---

## 💻 Cuplikan Alur Kerja Aplikasi

### 1. Mode Penjual (Input Mandiri & Pemindaian NLP)
- Penjual memilih merek (*Toyota*), model (*Avanza*), tahun (*2019*), transmisi (*Automatic*), odometer (*45.000 KM*), dan rencana harga (*Rp 160.000.000*).
- Penjual mengetik deskripsi iklan: *"Mobil terawat, servis rutin bengkel resmi, bebas banjir dan tabrakan."*
- **NLP Real-Time Scanner** mendeteksi fitur nilai tambah (`ada_garansi = Ya`).
- Model menghitung Nilai Wajar Pasar: **Rp 168.000.000** (Deviasi: **-4.8%**).
- **Vonis Sistem**: **[✓] CEPAT LAKU (< 15 Hari)**.
- **Price Drop Simulator**: Penjual dapat menggeser harga untuk melihat batas maksimal harga agar unit tetap berstatus cepat laku.

### 2. Mode Pembeli (Inspeksi Unit & Peringatan TDP)
- Pembeli memilih unit dari daftar marketplace OLX.
- Jika menemukan iklan unit tahun muda dengan harga Rp 35.000.000 yang mencantumkan *"TDP minim cicilan murah"*, sistem langsung menyalakan alarm merah:
  > ⚠️ **WASPADA INDIKASI HARGA JEBAKAN (TDP/KREDIT)**: Unit terindikasi skema uang muka leasing, bukan harga jual tunai.
- Jika unit wajar, sistem menampilkan batas rekomendasi negosiasi (*aggressive offer ceiling*) agar pembeli memperoleh kesepakatan terbaik (*Fair Deal*).

---

## 👥 Kontributor & Lisensi

Dikembangkan sebagai implementasi komprehensif **Sistem Pendukung Keputusan (Decision Support System) Otomotif** berbasis *Machine Learning*, *Data Mining*, dan *Natural Language Processing*.

Proyek ini didistribusikan di bawah lisensi **MIT License**. Silakan merujuk ke berkas [LICENSE](LICENSE) untuk informasi lebih detail.