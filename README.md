# AutoLaku

**Sistem Prediksi Kecepatan Laku Mobil Bekas OLX Berdasarkan Harga Wajar, Jarak Tempuh, dan Deskripsi Iklan Menggunakan Metode Klasifikasi**

---

## Ringkasan Proposal

| Komponen | Keterangan |
|---|---|
| **Nama Produk** | AutoLaku |
| **Masalah** | Penjual mobil bekas di OLX kesulitan menentukan apakah harga yang mereka pasang akan membuat mobilnya cepat terjual atau justru mengendap lama. Tidak ada alat bantu yang memberi umpan balik langsung berdasarkan data pasar nyata. |
| **Pengguna** | Penjual mobil bekas perseorangan yang ingin memperkirakan daya saing harga iklannya sebelum dipasang. |
| **Dataset** | Iklan mobil bekas yang di-scraping dari OLX Indonesia (`dataset_olx_mentah.csv`), berisi ribuan baris data iklan aktif. |
| **Fungsi Utama** | Menerima input spesifikasi mobil dan rencana harga jual, lalu mengeluarkan prediksi apakah mobil tersebut akan **Cepat**, **Sedang**, atau **Lambat** laku — disertai estimasi harga wajar pasar. |
| **Jenis Data Mining** | **Classification** (prediksi kategori kecepatan laku: Cepat / Sedang / Lambat), dengan regresi sebagai tahap awal untuk menghitung harga wajar yang menjadi fitur utama. |

---

## Pipeline Sistem

```
┌─────────────────────────────────────────────────────────────────┐
│  TAHAP 0 — SCRAPING  (scraping.py)                              │
│                                                                 │
│  • Buka halaman pencarian mobil bekas di OLX via browser        │
│    otomatis (Playwright/asyncio)                                │
│  • Intersepsi respons API OLX yang memuat data iklan (JSON)     │
│  • Ambil per iklan: judul, merek, model, tahun, transmisi,      │
│    harga, jarak_tempuh, tipe_penjual, deskripsi,                │
│    tanggal_posting, lokasi                                      │
│  • Scroll otomatis untuk memuat lebih banyak halaman            │
│  • Simpan semua hasil ke dataset_olx_mentah.csv                 │
│                                                                 │
│  Output → dataset_olx_mentah.csv                                │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  TAHAP 1 — PREPROCESSING  (processing.py)                       │
│                                                                 │
│  • Ekstraksi merek, model, tahun, transmisi dari judul iklan    │
│  • Filter harga boneka/placeholder (mis. 99999999, 88888888)    │
│  • Imputasi jarak tempuh yang kosong (median per tahun)         │
│  • Hitung usia_mobil dan km_per_tahun                           │
│  • Baca deskripsi → deteksi garansi & indikasi kredit/TDP       │
│    (termasuk mengenali negasi: "tidak ada garansi" = Tidak)     │
│  • Tandai kombinasi merek+model+usia yang datanya sedikit       │
│                                                                 │
│  Output → dataset_olx_bersih.csv                                │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  TAHAP 2 — PELATIHAN MODEL  (train_model_klasifikasi.py)        │
│                                                                 │
│  [2a] Model Regresi Harga Wajar                                 │
│       Input : 9 fitur kendaraan (merek, model, tahun, dll.)     │
│       Metode: Gradient Boosting Regressor (log-harga)           │
│               + Fallback regresi usia untuk sampel sedikit      │
│       Teknik: out-of-fold (harga wajar tidak bocor dari iklan   │
│               itu sendiri)                                      │
│       Output: harga_wajar_final per iklan                       │
│                                                                 │
│  [2b] Pembentukan Label Proxy                                   │
│       deviasi_persen = (harga_iklan − harga_wajar) / harga_wajar│
│       Bagi dengan tertile (Q33 & Q67 dari distribusi data):     │
│         ≤ Q33 → Cepat  |  Q33–Q67 → Sedang  |  > Q67 → Lambat │
│       Penyesuaian: km/tahun tinggi & tanpa garansi → ke Sedang  │
│                                                                 │
│  [2c] Model Klasifikasi Kecepatan Laku                          │
│       Input : 10 fitur (9 kendaraan + deviasi_persen)           │
│       Kandidat: Logistic Regression, Decision Tree,             │
│                 Random Forest, Gradient Boosting                │
│       Kriteria pilih: (1) monoton thd harga, (2) tidak          │
│                       overconfident, (3) F1-macro tertinggi     │
│       Output: Cepat / Sedang / Lambat + probabilitas per kelas  │
│                                                                 │
│  Tersimpan:                                                     │
│    model_regresi_harga.pkl                                      │
│    model_klasifikasi_likuiditas.pkl                             │
│    autolaku_meta.json                                           │
│    output_klasifikasi_<tanggal>/  (9 grafik evaluasi)           │
└─────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  TAHAP 3 — PREDIKSI  (mesin_prediksi.py)                        │
│                                                                 │
│  Input penjual: merek, model, tahun, transmisi, km,             │
│                 rencana harga jual, teks deskripsi iklan        │
│                                                                 │
│  Proses:                                                        │
│  1. Validasi & normalisasi semua input                          │
│  2. Hitung harga wajar (model regresi + fallback bila perlu)    │
│  3. Hitung deviasi_persen terhadap harga wajar                  │
│  4. Prediksi kelas likuiditas + probabilitas tiap kelas         │
│  5. Berikan peringatan bila data pembanding sedikit             │
│                                                                 │
│  Output: STATUS (Cepat / Sedang / Lambat)                       │
│          + Harga Wajar + Deviasi % + Probabilitas per kelas     │
└─────────────────────────────────────────────────────────────────┘
     │
     ├──► Terminal       (test_prediksi_klasifikasi.py)
     └──► Aplikasi Web   (app.py — Streamlit)
```

---

## Atribut yang Digunakan

### Atribut Mentah (dari scraping OLX — 10 kolom)

| Kolom | Tipe | Keterangan |
|---|---|---|
| `judul` | Teks | Judul iklan — dipakai untuk mengekstrak model bila kolom model kosong |
| `merek` | Teks | Merek kendaraan (Toyota, Honda, Daihatsu, dll.) |
| `model` | Teks | Model/seri (Avanza, Brio, dll.) — diverifikasi ulang dari judul |
| `tahun` | Angka | Tahun perakitan — diambil dari kolom atau diekstrak dari judul |
| `transmisi` | Teks | Jenis transmisi — dinormalisasi dari berbagai ejaan (matic/at/cvt → automatic) |
| `harga` | Angka | Harga iklan dalam Rupiah — harga boneka/placeholder otomatis disaring |
| `jarak_tempuh` | Angka | Odometer dalam KM — nilai rentang (mis. "50.000–60.000") dirata-rata |
| `deskripsi` | Teks | Deskripsi iklan — dibaca untuk mendeteksi garansi dan indikasi kredit |
| `tipe_penjual_badge` | Teks | Badge penjual dari OLX (Individu / Dealer) |
| `tanggal_posting` | Teks | Tanggal iklan tayang — hanya untuk validasi kasar label proxy |

### Atribut Turunan (dibuat saat preprocessing — 7 kolom baru)

| Kolom | Dibuat Dari | Penjelasan |
|---|---|---|
| `usia_mobil` | `tahun` | 2026 − tahun perakitan |
| `km_per_tahun` | `jarak_tempuh` ÷ `usia_mobil` | Rata-rata pemakaian per tahun — indikator intensitas pakai |
| `ada_garansi` | `deskripsi`, `judul` | "Ya" bila ada kata garansi/service record/bebas banjir **tanpa** didahului kata negasi |
| `indikasi_kredit_tdp` | `deskripsi`, `judul` | 1 bila deskripsi menyebut TDP/cicilan/over kredit |
| `tipe_penjual` | `tipe_penjual_badge` | Normalisasi: Individu atau Dealer |
| `usia_bracket` | `usia_mobil` | Kelompok usia per 5 tahun (0–4, 5–9, 10–14, dst.) |
| `jumlah_sampel_kategori` | `merek`, `model`, `usia_bracket` | Jumlah iklan sejenis — dasar peringatan keyakinan rendah (ambang: 15 data) |

### Fitur yang Masuk ke Model

**Model Regresi Harga Wajar — 9 fitur:**

| # | Fitur | Encoding |
|---|---|---|
| 1 | `merek` | One-Hot |
| 2 | `model` | One-Hot |
| 3 | `transmisi` | One-Hot |
| 4 | `tipe_penjual` | One-Hot |
| 5 | `ada_garansi` | One-Hot |
| 6 | `tahun` | StandardScaler |
| 7 | `jarak_tempuh` | StandardScaler |
| 8 | `usia_mobil` | StandardScaler |
| 9 | `km_per_tahun` | StandardScaler |

**Model Klasifikasi Kecepatan Laku — 10 fitur** (= 9 di atas + 1):

| # | Fitur Tambahan | Keterangan |
|---|---|---|
| 10 | `deviasi_persen` | Simpangan harga iklan terhadap harga wajar (%) — **fitur terpenting** |

> **Catatan penting:** Uji ablation membuktikan akurasi turun dari **99,1% → 53,8%** bila `deviasi_persen` dihapus. Ini menunjukkan sirkularitas — label dibentuk dari deviasi, lalu deviasi dipakai untuk memprediksinya. Akurasi tinggi tidak berarti sistem "terbukti cocok dengan realita pasar".

---

## Pembentukan Label (Target Klasifikasi)

Label **Cepat / Sedang / Lambat** adalah *proxy* — bukan data transaksi nyata — karena OLX tidak menyediakan informasi kapan iklan benar-benar terjual.

Cara pembentukan label:

1. Hitung `deviasi_persen` setiap iklan secara *out-of-fold* (model tidak melihat iklan itu sendiri saat menghitung harga wajarnya).
2. Bagi distribusi deviasi dengan **tertile** — nilai Q33 dan Q67 dihitung dari data, bukan ditentukan manual:
   - Nilai aktual (latihan 29 Sep 2026): **Q33 = −8,07%**, **Q67 = +7,67%**
   - Deviasi ≤ −8,07% → **Cepat** | −8,07% s.d. +7,67% → **Sedang** | > +7,67% → **Lambat**
3. Unit dengan km/tahun > 30.000 **dan** tanpa garansi yang masuk kelas Cepat, diturunkan ke **Sedang**.

---

## Pemilihan Model Klasifikasi

Empat algoritma diuji dan dibandingkan sebelum satu dipilih sebagai model final:

| Algoritma | Catatan |
|---|---|
| Logistic Regression | Sederhana, cepat, probabilitas terkalibrasi |
| Decision Tree | Mudah diinterpretasi, tapi rentan overfit dan overconfident |
| Random Forest | Ensemble — lebih stabil, probabilitas lebih wajar |
| Gradient Boosting | Akurasi tinggi, lebih lambat dilatih |

**Kriteria pemilihan (berurutan, semua harus dipenuhi):**
1. **Monoton terhadap harga** — harga naik tidak boleh membuat prediksi jadi lebih cepat laku. Ini syarat mutlak.
2. **Tidak overconfident** — rata-rata keyakinan prediksi tidak boleh > 0,97. Decision Tree tunggal sering menghasilkan probabilitas 100% per daun, yang merupakan artefak algoritma, bukan tanda benar-benar yakin.
3. Di antara yang lolos dua syarat di atas, dipilih yang **F1-macro tertinggi**.

**Model terpilih (latihan 29 Sep 2026): Random Forest** — keyakinan rata-rata 0,692, RMSE ordinal 0,093.

---

## Struktur File

| File | Fungsi |
|---|---|
| `scraping.py` | Mengambil data iklan dari OLX dengan intersepsi API, disimpan ke `dataset_olx_mentah.csv` |
| `kamus_slang_otomotif.py` | Kamus normalisasi singkatan dan slang otomotif yang dipakai saat membersihkan teks |
| `processing.py` | Membersihkan data mentah, mengekstrak semua atribut, menghasilkan `dataset_olx_bersih.csv` |
| `train_model_klasifikasi.py` | Melatih model regresi harga wajar dan classifier likuiditas, menyimpan `.pkl` + grafik evaluasi |
| `mesin_prediksi.py` | Satu fungsi prediksi terpusat yang dipakai oleh CLI, tes otomatis, dan aplikasi web |
| `test_prediksi_klasifikasi.py` | Antarmuka terminal — penjual input spesifikasi mobil, sistem keluarkan hasil prediksi |
| `test_sistem.py` | Tes otomatis: validasi input, deteksi negasi pada deskripsi, uji monotonisitas harga |
| `app.py` | Aplikasi web Streamlit dengan dua mode: Penjual (input mandiri) dan Pembeli (eksplorasi dataset) |

File yang dihasilkan otomatis (jangan diedit manual):
`dataset_olx_bersih.csv`, `model_regresi_harga.pkl`, `model_klasifikasi_likuiditas.pkl`, `autolaku_meta.json`, folder `output_klasifikasi_<tanggal>/`.

---

## Cara Menjalankan

Semua file harus ada dalam satu folder, termasuk `dataset_olx_mentah.csv` hasil scraping.

```bash
pip install pandas numpy scikit-learn matplotlib scipy streamlit

python processing.py                  # 1. Bersihkan data mentah → bersih.csv
python train_model_klasifikasi.py     # 2. Latih model → simpan .pkl & grafik
python test_sistem.py                 # 3. Jalankan tes otomatis (opsional tapi disarankan)
python test_prediksi_klasifikasi.py   # 4. Coba prediksi via terminal
streamlit run app.py                  # 5. Buka aplikasi web di browser
```

Urutan ini wajib — setiap langkah membutuhkan hasil dari langkah sebelumnya. Jika `dataset_olx_mentah.csv` diperbarui atau ada perubahan kode di `processing.py` / `train_model_klasifikasi.py`, ulangi dari langkah 1.

---

## Metrik Evaluasi (Latihan Terakhir: 29 Sep 2026)

### Model Harga Wajar (Regresi)

| Metrik | Nilai |
|---|---|
| MAPE | 19,49% |
| MAE | Rp 81.606.098 |
| RMSE | Rp 197.722.509 |

### Model Klasifikasi (Random Forest)

| Metrik | Nilai | Keterangan |
|---|---|---|
| Accuracy (dengan `deviasi_persen`) | 99,14% | Sebagian besar bersifat sirkular (lihat ablation) |
| Accuracy (tanpa `deviasi_persen`) | 53,75% | Kemampuan prediksi tanpa fitur utama |
| Selisih ablation | −45,4 poin | Menunjukkan dominasi fitur deviasi |
| RMSE Ordinal | 0,093 | Galat peringkat kelas (0 = sempurna, >1 = meleset >1 kelas) |
| Rerata Keyakinan Puncak | 0,692 | Tidak overconfident |

### Grafik Evaluasi (di folder `output_klasifikasi_<tanggal>/`)

| File | Isi |
|---|---|
| `1_komparasi_metrik_klasifikasi.png` | Accuracy & F1-Macro keempat algoritma berdampingan |
| `2_confusion_matrix.png` | Confusion matrix model terpilih |
| `3_distribusi_label_proxy.png` | Jumlah data per kelas Cepat / Sedang / Lambat |
| `4_feature_importance_klasifikasi.png` | 15 fitur terpenting model terpilih |
| `5_akurasi_per_kecukupan_sampel.png` | Akurasi dibedakan: data sedikit vs. data cukup |
| `6_umur_iklan_per_label.png` | Validasi kasar: sebaran lama iklan tayang per kelas |
| `7_mae_rmse_harga_wajar.png` | Galat model regresi harga wajar dalam juta Rupiah |
| `8_rmse_ordinal_classifier.png` | RMSE ordinal semua algoritma sebagai pembanding |
| `9_deviasi_per_label.png` | Sebaran deviasi harga per kelas — cek monotonisitas label |

---

## Keterbatasan

1. **Label bukan data terjual sungguhan.** OLX tidak menyediakan status terjual atau tanggal iklan turun, sehingga label Cepat/Sedang/Lambat dibentuk dari deviasi harga, bukan observasi transaksi nyata.
2. **Akurasi tinggi bersifat sirkular.** `deviasi_persen` sekaligus dipakai membentuk label dan memprediksinya. Tanpa fitur ini, akurasi turun 45,4 poin — menandakan classifier sebagian besar hanya "menghafal" aturan tertile.
3. **Validasi kasar dengan umur iklan lemah.** Korelasi Spearman antara deviasi dan lama iklan tayang hanya ρ ≈ 0,06. Ini bukan bukti sistem akurat, hanya indikasi arah yang benar.
4. **Merek dan model terbatas** pada daftar di `MODEL_PER_MEREK` (`processing.py`). Unit di luar daftar jatuh ke kategori "Lainnya" dan prediksinya kurang spesifik.
5. **Prediksi keyakinan rendah untuk unit langka.** Kombinasi merek+model+rentang usia dengan data < 15 unit diberi peringatan otomatis, dan harga wajarnya dihitung dengan metode fallback (regresi usia atau median), bukan model ML utama.
6. **Data bersifat snapshot.** Tidak ada pembaruan otomatis — perlu scraping ulang secara berkala agar harga wajar tetap mencerminkan kondisi pasar terkini.