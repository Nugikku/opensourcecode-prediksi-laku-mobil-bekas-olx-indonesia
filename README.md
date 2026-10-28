# AutoLaku 🚗

**Sistem Prediksi Kecepatan Laku Mobil Bekas OLX Berdasarkan Harga Wajar, Jarak Tempuh, dan Deskripsi Iklan Menggunakan Metode Klasifikasi**

Tugas Data Mining – Kelas TI24A2 | Universitas Duta Bangsa Surakarta

---

## Deskripsi

AutoLaku membantu penjual perorangan mobil bekas di OLX memperkirakan apakah rencana harga jual mereka akan membuat unit **Cepat**, **Sedang**, atau **Lambat** laku — sebelum iklan dipasang.

Sistem mengambil data iklan dari OLX, menghitung harga wajar pasar, lalu mengklasifikasikan kecepatan laku berdasarkan selisih harga, jarak tempuh, dan isi deskripsi iklan.

> **Catatan metodologi:** Label Cepat/Sedang/Lambat bersifat *label proxy* — dibentuk dari deviasi harga iklan terhadap harga wajar, bukan dari data status terjual sesungguhnya (OLX tidak menyediakan data ini). Lihat bagian [Keterbatasan](#keterbatasan) untuk penjelasan lengkap.

---

## Alur Pipeline

```
scraping.py              → dataset_olx_mentah.csv       (3.000 baris)
processing.py            → dataset_olx_bersih.csv       (2.334 baris setelah pembersihan)
pelabelan.py             → dataset_olx_berlabel.csv     (split train/test + label proxy)
analisis_fitur.py        → fitur_final.json             (feature selection)
train_model_klasifikasi.py → model_*.pkl + autolaku_meta.json
test_sistem.py           → 21 tes otomatis
app.py / test_prediksi_klasifikasi.py → antarmuka pengguna
```

---

## Struktur File

| File | Peran |
|---|---|
| `scraping.py` | Scraping iklan mobil bekas dari OLX |
| `kamus_slang_otomotif.py` | Kamus normalisasi istilah otomotif |
| `processing.py` | Pembersihan data mentah → `dataset_olx_bersih.csv` |
| `pelabelan.py` | Split train/test, model harga wajar, label proxy |
| `analisis_fitur.py` | Feature selection → `fitur_final.json` |
| `model_khusus.py` | Implementasi Naive Bayes Diskretisasi (P9) |
| `train_model_klasifikasi.py` | Perbandingan, tuning, evaluasi model klasifikasi |
| `mesin_prediksi.py` | Engine prediksi (dipakai CLI, tes, dan Streamlit) |
| `test_prediksi_klasifikasi.py` | Antarmuka terminal untuk penjual |
| `test_sistem.py` | 21 tes otomatis |
| `app.py` | Antarmuka Streamlit |

**File yang dihasilkan otomatis:**
`dataset_olx_bersih.csv`, `dataset_olx_berlabel.csv`, `fitur_final.json`,
`model_regresi_harga.pkl`, `model_klasifikasi_likuiditas.pkl`, `autolaku_meta.json`,
`keputusan_fitur.csv`, `laporan_preprocessing.txt`

---

## Cara Menjalankan

### 1. Persyaratan

```bash
pip install pandas numpy scikit-learn matplotlib streamlit
```

### 2. Urutan eksekusi

```bash
# Mulai dari data mentah:
python processing.py
python pelabelan.py
python analisis_fitur.py
python train_model_klasifikasi.py
python test_sistem.py             # harus 21/21 lulus

# Antarmuka:
streamlit run app.py              # antarmuka web
python test_prediksi_klasifikasi.py  # antarmuka terminal
```

> **Penting:** Hapus `.pkl` dan `autolaku_meta.json` lama sebelum menjalankan ulang dari awal. Jangan lewati `analisis_fitur.py` — hasil `fitur_final.json`-nya dibaca oleh `train_model_klasifikasi.py`.

### 3. Jika hanya ingin menjalankan aplikasi (tanpa training ulang)

Pastikan semua file `.pkl`, `autolaku_meta.json`, `dataset_olx_berlabel.csv`, dan `fitur_final.json` sudah tersedia di folder yang sama, lalu:

```bash
streamlit run app.py
```

---

## Data

- **Sumber:** Scraping mandiri dari [OLX Indonesia](https://www.olx.co.id/mobil-bekas_c198)
- **Data mentah:** 3.000 baris, 13 atribut
- **Data bersih:** 2.334 baris setelah 12 tahap preprocessing
- **Split:** 1.859 data training (80%) | 465 data test (20%), stratifikasi per kuartil harga

### Tahap preprocessing utama (P3-4)

| Tahap | Hasil |
|---|---|
| Ekstraksi tahun, transmisi, model dari judul | Menutup nilai kosong efektif di kolom mentah |
| Standardisasi merek (51 → 42 varian) | Penulisan tidak konsisten diseragamkan |
| Pembersihan harga (filter boneka, luar rentang) | 3.000 → 2.660 baris |
| Deteksi noise berbasis aturan domain (km) | 10 baris ditandai & dikeluarkan dari model |
| Deteksi outlier IQR log-harga per merek | 74 baris **ditandai** (tidak dibuang — mobil mewah bisa valid) |
| Fitur teks diperluas dengan penanganan negasi | `ada_garansi`, `riwayat_negatif`, `catatan_servis`, `pajak_hidup` |
| Hapus duplikat | 2.638 → 2.334 baris |
| Penanda transmisi asumsi | 1.021 baris (34%) tidak menyebut transmisi di iklan, diisi 'automatic' dan ditandai |

---

## Feature Selection (P3-4)

Dilakukan di `analisis_fitur.py` sebelum training. Hasilnya tersimpan di `fitur_final.json` dan dibaca otomatis oleh `train_model_klasifikasi.py`.

| Metode | Fitur paling berpengaruh |
|---|---|
| Korelasi (filter) | `deviasi_persen`, `tahun`, `km_per_tahun`, `jarak_tempuh` |
| Mutual information | `deviasi_persen`, `merek`, `model`, `tahun` |
| Random Forest importance (embedded) | `deviasi_persen` dominan; `tahun`, `km_per_tahun` berikutnya |

**Fitur dibuang:**
- `tipe_penjual` — konstan (seluruh data penjual individu)
- `usia_mobil` — redundan dengan `tahun` (r = 0,9995)

**Fitur final yang dipakai model klasifikasi (11 fitur):**
`merek`, `model`, `transmisi`, `ada_garansi`, `riwayat_negatif`, `catatan_servis`, `pajak_hidup`, `tahun`, `jarak_tempuh`, `km_per_tahun`, `deviasi_persen`

---

## Model

### Harga Wajar (P5)

Model regresi untuk menghitung harga wajar tiap unit, sebelum deviasi dihitung.

| Aspek | Detail |
|---|---|
| Model terpilih | **Ridge Regression** (alpha=1.0) |
| Kriteria pilih | MAPE terendah pada 5-fold CV data training |
| Target | log(harga) → dikembalikan ke Rupiah |
| MAPE | **18,77%** |
| R² | **0,7726** |
| RMSE | Rp 190,4 juta |
| Kandidat dibandingkan | Linear Regression, Ridge, Random Forest, Gradient Boosting |

Untuk kombinasi merek/model/usia dengan data sedikit (< 15 sampel), dipakai fallback regresi usia per model atau per merek.

### Klasifikasi Likuiditas (P6, P9, P12, P13)

| Aspek | Detail |
|---|---|
| Model terpilih | **Random Forest** |
| Kriteria pilih | F1-macro CV tertinggi di antara model yang lolos: monoton terhadap harga, tidak overconfident (keyakinan ≤ 0,97), selisih train–validasi < 10% |
| F1-macro CV | **0,9941 ± 0,0046** (5-fold Stratified) |
| F1-macro test | **0,9872** |
| Akurasi test | **0,9871** |
| RMSE ordinal test | **0,1136** |
| Rerata keyakinan puncak | **0,7213** |

**Kandidat yang dibandingkan (Skenario 2, dengan `deviasi_persen`):**

| Model | F1 CV | Std | F1 Train | Keputusan |
|---|---|---|---|---|
| Logistic Regression | 0,9384 | 0,0154 | 0,9729 | OK |
| KNN (K=5) | 0,6159 | 0,0115 | 0,7693 | Ditolak: overfitting |
| Naive Bayes (Bernoulli) | 0,8762 | 0,0085 | 0,8928 | OK |
| Decision Tree | 0,9979 | 0,0020 | 1,0000 | Ditolak: overconfident |
| **Random Forest** | **0,9941** | 0,0046 | 1,0000 | **Terpilih** |
| Gradient Boosting | 0,9979 | 0,0020 | 1,0000 | Ditolak: overconfident |

**Tuning Hyperparameter + Nested CV (P13):**
RandomizedSearchCV (n_iter=8, inner CV=3) pada model terpilih. F1 nested CV = 0,9957 ± 0,0036. Selisih dengan default (0,9941) tidak melebihi satu standar deviasi → **parameter default dipertahankan**.

**Label Proxy:**

| Label | Kondisi |
|---|---|
| Cepat | Deviasi harga ≤ −6,96% (Q33 dari data training) |
| Sedang | −6,96% < deviasi ≤ 7,30% (Q33–Q67) |
| Lambat | Deviasi harga > 7,30% (Q67 dari data training) |

Ambang hanya dihitung dari data **training** untuk menghindari data leakage (P12).

### Uji Ablation

| Skenario | Akurasi | Keterangan |
|---|---|---|
| Dengan `deviasi_persen` | **0,9871** | Model yang dipakai aplikasi |
| Tanpa `deviasi_persen` | **0,5140** | Mendekati tebakan acak (1/3) |

Akurasi tinggi sebagian bersifat sirkular — classifier mempelajari ulang aturan yang membentuk labelnya sendiri. Skenario 1 (tanpa deviasi) adalah perbandingan algoritma yang lebih adil.

---

## Evaluasi Model (P12)

- **Split data:** train/test dipisah **sebelum** label dibuat dan sebelum ambang tertile dihitung
- **Preprocessing dalam pipeline CV:** StandardScaler dan OneHotEncoder hanya di-fit dari data training di tiap fold
- **Test set dibuka sekali:** hanya untuk evaluasi akhir, tidak dipakai memilih model atau parameter
- **Cek overfitting:** selisih F1 training vs validasi dilaporkan untuk setiap kandidat
- **Tes otomatis:** 21 tes, termasuk uji monotonisitas, kebocoran data, konsistensi fitur, dan fitur teks baru

---

## Keterbatasan

1. **Label proxy** — OLX tidak menyediakan data status terjual; label dibentuk dari deviasi harga
2. **Akurasi tinggi sebagian sirkular** — lihat uji ablation; akurasi tanpa `deviasi_persen` hanya 51%
3. **Data snapshot** — bukan real-time; harga pasar bisa berubah
4. **Transmisi 34% asumsi** — iklan yang tidak menyebut transmisi diisi 'automatic' (ditandai `transmisi_dari_teks=0`)
5. **Validasi umur iklan tidak reliabel** — `tanggal_posting` kemungkinan tanggal dibuat, bukan diperbarui
6. **Fokus penjual saja** — tidak ada mode pembeli

---

## Lisensi

Proyek ini dibuat untuk keperluan akademik (Tugas Data Mining, TI24A2). Data bersumber dari scraping publik OLX.