# AutoLaku 🚗

**Sistem Prediksi Kecepatan Laku Mobil Bekas OLX Berdasarkan Harga Wajar, Jarak Tempuh, dan Deskripsi Iklan Menggunakan Metode Klasifikasi**

Tugas Data Mining – Kelas TI24A2 | Universitas Duta Bangsa Surakarta

---

## Deskripsi

AutoLaku membantu penjual perorangan mobil bekas di OLX memperkirakan apakah rencana harga jual membuat unit **Cepat**, **Sedang**, atau **Lambat** laku, sebelum iklan dipasang.

Sistem menghitung harga wajar dari data iklan OLX, mengukur seberapa jauh rencana harga dari harga wajar itu, lalu mengklasifikasikan kecepatan laku bersama jarak tempuh dan isi deskripsi iklan.

> **Catatan metodologi:** Label Cepat/Sedang/Lambat adalah *label proxy*. OLX tidak menyediakan data status terjual, sehingga label dibentuk dari deviasi harga iklan terhadap harga wajar. Dasar ilmiahnya ada di bagian [Landasan Ilmiah Pelabelan](#landasan-ilmiah-pelabelan), keterbatasannya di bagian [Keterbatasan](#keterbatasan).

---

## Alur Pipeline

```
scraping.py                → dataset_olx_mentah.csv
processing.py              → dataset_olx_bersih.csv
pelabelan.py               → dataset_olx_berlabel.csv   (split train/test + label proxy)
analisis_fitur.py          → fitur_final.json           (feature selection)
train_model_klasifikasi.py → model_*.pkl + autolaku_meta.json
test_sistem.py             → 21 tes otomatis
app.py / test_prediksi_klasifikasi.py → antarmuka pengguna
```

## Struktur File

| File | Peran |
|---|---|
| `scraping.py` | Scraping iklan mobil bekas dari OLX |
| `kamus_slang_otomotif.py` | Kamus normalisasi istilah otomotif |
| `processing.py` | Pembersihan data mentah |
| `pelabelan.py` | Split train/test, model harga wajar, label proxy |
| `analisis_fitur.py` | Feature selection → `fitur_final.json` |
| `model_khusus.py` | Naive Bayes dengan diskretisasi fitur numerik |
| `train_model_klasifikasi.py` | Perbandingan, tuning, dan evaluasi model klasifikasi |
| `tambah_data.py` | Menambah data latih baru lalu melatih ulang otomatis (dengan arsip dan pemulihan) |
| `mesin_prediksi.py` | Engine prediksi (dipakai CLI, tes, dan Streamlit) |
| `test_prediksi_klasifikasi.py` | Antarmuka terminal untuk penjual |
| `test_sistem.py` | 21 tes otomatis |
| `app.py` | Antarmuka Streamlit |

File yang dihasilkan otomatis: `dataset_olx_bersih.csv`, `dataset_olx_berlabel.csv`, `fitur_final.json`, `model_regresi_harga.pkl`, `model_klasifikasi_likuiditas.pkl`, `autolaku_meta.json`, `keputusan_fitur.csv`, `laporan_preprocessing.txt`.

---

## Cara Menjalankan

```bash
pip install pandas numpy scikit-learn matplotlib streamlit

python processing.py
python pelabelan.py
python analisis_fitur.py
python train_model_klasifikasi.py     # beberapa menit (nested CV)
python test_sistem.py                 # target: 21/21 lulus

streamlit run app.py                  # antarmuka web
python test_prediksi_klasifikasi.py   # antarmuka terminal
```

> Hapus `.pkl` dan `autolaku_meta.json` lama sebelum menjalankan ulang dari awal. Jangan lewati `analisis_fitur.py`, karena `train_model_klasifikasi.py` membaca `fitur_final.json`.
> Jika menggabungkan beberapa batch scraping, gabungkan CSV mentah lebih dulu. `processing.py` membuang iklan ganda lewat `id_iklan`.

---

## Menambah Data Latih Baru

Sistem bisa menerima data latih baru kapan saja. Model scikit-learn tidak belajar bertahap, jadi data baru digabung ke dataset lama lalu seluruh pipeline dijalankan ulang.

**Lewat terminal:**

```bash
python tambah_data.py hasil_scraping_baru.csv
python tambah_data.py hasil_scraping_baru.csv --tanpa-latih   # hanya gabung data
python tambah_data.py hasil_scraping_baru.csv --paksa         # tetap simpan walau tes gagal
```

**Lewat aplikasi:** buka sidebar `streamlit run app.py`, pilih **Admin: Tambah Data Latih**, unggah CSV, lalu klik *Tambahkan & latih ulang*.

Yang dilakukan sistem:
1. Memeriksa kolom CSV baru (harus sama dengan `dataset_olx_mentah.csv`) dan membuang iklan yang `id_iklan`-nya sudah ada
2. Mengarsipkan data dan model lama di folder `arsip/<waktu>/`
3. Menjalankan `processing.py`, `pelabelan.py`, `analisis_fitur.py`, `train_model_klasifikasi.py`, lalu `test_sistem.py`
4. Jika ada tahap yang gagal, semua file dikembalikan otomatis dari arsip
5. Mencatat riwayat (jumlah data, F1, akurasi) di `riwayat_data.json` dan menampilkan perbandingan sebelum dan sesudah

Catatan: penamaan model antar-batch belum diseragamkan otomatis (contoh `Innova` dan `Kijang Innova`), dan motor atau truk yang ikut terscraping belum difilter. Periksa `laporan_preprocessing.txt` setelah menambah data. Tambahkan `arsip/` ke `.gitignore`.

---

## Hasil per Batch Data

Pipeline dijalankan pada dua batch hasil scraping. **Batch 1** adalah data awal (3.000 baris mentah). **Batch 2** adalah hasil scraping ulang. Angka di bawah dihitung dari `dataset_olx_bersih.csv` dan `dataset_olx_berlabel.csv` masing-masing batch, pada data test.

### Ukuran data dan pelabelan

| Aspek | Batch 1 | Batch 2 |
|---|---|---|
| Data mentah | 3.000 baris | lihat `laporan_preprocessing.txt` |
| Data bersih | 2.334 | 2.643 |
| Noise km ditandai | 10 | 1 |
| Data berlabel | 2.324 | 2.642 |
| Train / Test | 1.859 / 465 | 2.113 / 529 |
| Merek unik | 42 | 56 |
| Model unik | 151 | 396 |
| Baris bermodel "Lainnya" | 15,2% | 0% |
| Kombinasi dengan sampel < 15 | 61,6% | 66,9% |
| Transmisi hanya asumsi | 31,1% | 35,4% |
| Harga median | Rp 220 jt | Rp 210 jt |
| Ambang Q33 / Q67 (dari train) | −6,96% / +7,30% | −7,27% / +6,51% |

Ambang tertile hampir sama di kedua batch (sekitar ±7%), jadi batas Cepat/Sedang/Lambat relatif stabil terhadap penambahan data.

### Distribusi label proxy (%)

| Data | Cepat | Sedang | Lambat |
|---|---|---|---|
| Batch 1 – train | 32,8 | 34,2 | 33,0 |
| Batch 1 – test | 34,0 | 34,2 | 31,8 |
| Batch 2 – train | 32,7 | 34,3 | 33,0 |
| Batch 2 – test | 28,9 | 33,8 | 37,2 |

Distribusi test tidak sama dengan 33/33/33 karena ambang dihitung hanya dari train. Pada batch 2, test condong ke Lambat (37,2%). Ini yang terjadi jika ambang tidak ikut dilihat oleh data test.

### Model harga wajar (data test)

| Metrik | Batch 1 | Batch 2 |
|---|---|---|
| MAPE (model saja) | 19,80% | 18,48% |
| R² (model saja) | 0,7420 | 0,8462 |
| MAPE (model + fallback usia) | 18,43% | 18,16% |
| R² (model + fallback usia) | 0,7587 | 0,7896 |
| MAE (model + fallback usia) | Rp 76,5 jt | Rp 66,7 jt |
| Harga wajar dari model ML | 35,8% | 36,6% |
| Harga wajar dari fallback regresi usia (merek) | 8,2% | 17,9% |

Data yang lebih banyak memperbaiki model harga wajar. Pada batch 2, fallback usia menurunkan MAPE sedikit tetapi **menurunkan R²** (0,846 menjadi 0,790), sehingga ambang fallback perlu dievaluasi ulang.

### Hasil klasifikasi

| Metrik | Batch 1 | Batch 2 |
|---|---|---|
| Model terpilih | Random Forest | menyusul setelah training |
| F1-macro CV (5-fold) | 99,41% ± 0,46% | menyusul |
| F1-macro test | 98,72% | menyusul |
| Akurasi test | 98,71% | menyusul |
| Akurasi tanpa `deviasi_persen` (ablation) | 51,40% | menyusul |

Hasil training batch 2 belum dilampirkan. Jalankan `train_model_klasifikasi.py` lalu isi kolom ini.

### Tes otomatis

| | Batch 1 | Batch 2 |
|---|---|---|
| Hasil | 21/21 lulus | 20/21 sebelum perbaikan |

Satu tes gagal pada batch 2 karena kombinasi `Toyota Innova` tidak ditemukan. Penamaan model berubah antar-batch (batch 1: `Innova`, batch 2: `Kijang Innova`), sedangkan tes menulis nama model secara manual. Ini bukan bug sistem. Tes sekarang mengambil kasus dari data sehingga tidak bergantung pada nama model tertentu.

### Temuan pada data batch 2 yang perlu ditangani

1. **Model terlalu rinci.** Kolom `model` kini terisi penuh (nama seperti `218I`, `318I`, `A140`, `A200`), sehingga 285 dari 396 model hanya punya kurang dari 5 baris. Akibatnya sampel per kombinasi menipis dan lebih banyak harga wajar jatuh ke fallback tingkat merek.
2. **Kendaraan non-mobil ikut terscraping.** Ada 30 baris (1,1%) motor, truk, dan van: Yamaha, Kawasaki, KTM, Ducati, Harley Davidson, Piaggio, Aprilia, Hino, Maxus, dan "Truk". Perlu difilter berdasarkan daftar merek mobil.
3. **Standardisasi merek belum tuntas.** `Dfsk` dan `Dfsk (Dongfeng Sokon)` masih terpisah.
4. **Batch 2 bukan scraping ulang yang lengkap.** Hanya 185 `id_iklan` yang sama dengan batch 1. Karena itu iklan yang "hilang" di batch 2 tidak bisa dianggap sudah terjual.

---

## Data dan Preprocessing

- **Sumber:** scraping mandiri dari [OLX Indonesia](https://www.olx.co.id/mobil-bekas_c198)
- **Alur batch 1:** 3.000 mentah → 2.334 bersih → 2.324 berlabel (10 baris noise km dikeluarkan di `pelabelan.py`) → 1.859 train / 465 test (80/20, stratifikasi per kuartil harga)

| Tahap | Keterangan |
|---|---|
| Ekstraksi tahun, transmisi, model dari judul | Kolom mentah tahun/transmisi kosong efektif |
| Standardisasi merek | Penulisan tidak konsisten diseragamkan |
| Filter harga (boneka, di luar Rp 25 jt–3 miliar) dan tahun (1995–2026) | Data tidak wajar dibuang |
| Deteksi noise berbasis aturan domain (km) | Ditandai, lalu dikeluarkan saat pelabelan |
| Deteksi outlier IQR pada log-harga per merek | Hanya ditandai, tidak dibuang (mobil mewah bisa valid) |
| Fitur teks dengan penanganan negasi | `ada_garansi`, `riwayat_negatif`, `catatan_servis`, `pajak_hidup` |
| Hapus duplikat (batch 1: 304 baris) | Sebagian besar iklan yang dipasang ulang (66% judul identik, 83% lokasi sama) |
| Penanda transmisi asumsi | Iklan tanpa info transmisi diisi `automatic` dan ditandai `transmisi_dari_teks=0` |

---

## Feature Selection

Dilakukan di `analisis_fitur.py`, hanya pada data train. Hasilnya di `fitur_final.json` dan dibaca otomatis oleh training.

- **Dibuang:** `tipe_penjual` (konstan, seluruhnya penjual individu) dan `usia_mobil` (redundan dengan `tahun`, r = 0,9995)
- **Dipakai (11 fitur):** `merek`, `model`, `transmisi`, `ada_garansi`, `riwayat_negatif`, `catatan_servis`, `pajak_hidup`, `tahun`, `jarak_tempuh`, `km_per_tahun`, `deviasi_persen`
- **Temuan (batch 1):** `deviasi_persen` sangat dominan (importance Random Forest 64,79%). Fitur lain hampir tidak berpengaruh pada permutation importance.

---

## Model

### Harga wajar

Ridge Regression, target log(harga), dipilih dengan MAPE terendah pada 5-fold CV data train. Kandidat: Linear, Ridge, Random Forest, Gradient Boosting. Kombinasi dengan sampel kurang dari 15 memakai fallback regresi usia per model atau per merek.

### Klasifikasi (hasil final batch 1)

| Model | F1 CV | Std | F1 Train | Keputusan |
|---|---|---|---|---|
| Logistic Regression | 93,84% | 1,54% | 97,29% | OK |
| KNN (K=5) | 61,59% | 1,15% | 76,93% | Ditolak: overfitting |
| Naive Bayes (Bernoulli, diskretisasi) | 87,62% | 0,85% | 89,28% | OK |
| Decision Tree | 99,79% | 0,20% | 100% | Ditolak: overconfident |
| **Random Forest** | **99,41%** | 0,46% | 100% | **Terpilih** |
| Gradient Boosting | 99,79% | 0,20% | 100% | Ditolak: overconfident |

Kriteria pilih: F1-macro CV tertinggi di antara model yang monoton terhadap harga, tidak overconfident (keyakinan ≤ 0,97), dan tanpa overfitting berlebih. **Tuning:** RandomizedSearchCV dengan nested CV memberi F1 99,57% ± 0,36%. Selisih dengan parameter default tidak melebihi satu standar deviasi, sehingga parameter default dipertahankan.

**Label proxy:** Cepat jika deviasi ≤ Q33, Sedang jika di antara Q33 dan Q67, Lambat jika di atas Q67. Ambang hanya dari data train.

**Uji ablation:** tanpa `deviasi_persen`, akurasi turun dari 98,71% ke 51,40%, mendekati tebakan acak. Classifier sebagian besar mempelajari ulang aturan yang membentuk labelnya.

---

## Landasan Ilmiah Pelabelan

Label dibuat otomatis lewat kode, jadi setiap langkahnya perlu rujukan. Lima rujukan di bawah sudah dibaca isinya. Tiap rujukan dipakai untuk satu pertanyaan saja.

| Langkah di `pelabelan.py` | Rujukan | Alasan jadi patokan | Batasan |
|---|---|---|---|
| Deviasi harga mewakili kecepatan laku | **Granát & Vékás (2025)** | Pada marketplace mobil bekas, rasio harga iklan terhadap harga prediksi ML adalah prediktor kuat durasi iklan. Alurnya sama dengan AutoLaku, dan mereka menyebut alur ini bisa dipasang di marketplace untuk memberi tahu penjual perkiraan waktu jual. | Naskah, status terbit belum terverifikasi. Durasi = iklan menghilang (proksi penjualan). Data Hungaria. |
| Deviasi harga mewakili kecepatan laku | **Born dkk. (2018)** | Variabel DOP (harga ÷ harga hedonik log-linear) terbukti: makin tinggi DOP, makin lama mobil tidak terjual. Memakai data penjualan nyata. | *Working paper*. Data dealer Jerman. |
| Harga wajar dari regresi karakteristik | **Rosen (1974)** | Dasar teori harga hedonik: harga produk diregresikan pada karakteristiknya. | Hanya untuk harga wajar, tidak membahas waktu jual. Mengabaikan pasar barang bekas. |
| Label dibuat otomatis dari aturan | **Ratner dkk. (2017)**, Snorkel | Label dari aturan heuristik (*weak supervision*) adalah praktik yang diakui. | AutoLaku memakai satu aturan dan tidak memodelkan akurasinya, jadi bukan implementasi Snorkel. |
| Harga di atas pasar memperlambat penjualan | **Anglin dkk. (2003)** | Kenaikan harga listing menaikkan perkiraan waktu di pasar (TOM). Pendukung dari pasar properti. | Objeknya rumah. Baru abstrak yang dibaca. |

Ambang −6,96% dan +7,30% (batch 1) adalah persentil ke-33 dan ke-67 data train. Itu keputusan desain, bukan batas yang dinyatakan jurnal. Jurnal hanya mendukung **urutan** hubungannya (makin mahal, makin lama laku). Pada batch 2 ambangnya hampir sama (−7,27% dan +6,51%).

### Daftar Pustaka

- Granát, M. P., & Vékás, P. (2025). *Estimating Liquidity in Online Marketplaces via Machine Learning and Survival Analysis*. Naskah. https://real.mtak.hu/247298/1/epj_survival.pdf
- Born, A., Kovachka, N., Lessmann, S., & Seow, H.-V. (2018). *Price Management in the Used-Car Market: An Evaluation of Survival Analysis*. IRTG 1792 Discussion Paper No. 2018-065, Humboldt-Universität zu Berlin. https://hdl.handle.net/10419/230775
- Rosen, S. (1974). Hedonic Prices and Implicit Markets: Product Differentiation in Pure Competition. *Journal of Political Economy*, 82(1), 34–55.
- Ratner, A., Bach, S. H., Ehrenberg, H., Fries, J., Wu, S., & Ré, C. (2017). Snorkel: Rapid Training Data Creation with Weak Supervision. *Proceedings of the VLDB Endowment*, 11(3), 269–282. https://doi.org/10.14778/3157794.3157797
- Anglin, P. M., Rutherford, R., & Springer, T. M. (2003). The Trade-off Between the Selling Price of Residential Properties and Time-on-the-Market: The Impact of Price Setting. *Journal of Real Estate Finance and Economics*, 26(1), 95–111. https://doi.org/10.1023/A:1021526332732

---

## Evaluasi Model

- Split train/test dilakukan **sebelum** label dibuat dan sebelum ambang tertile dihitung
- Preprocessing (scaling, one-hot) berada di dalam pipeline CV, di-*fit* hanya dari data train tiap fold
- Test set dibuka sekali, hanya untuk evaluasi akhir
- Selisih F1 train dan validasi dilaporkan untuk tiap kandidat
- Referensi prediksi di aplikasi hanya data train
- Output metrik klasifikasi tampil dalam persen. RMSE ordinal tetap angka biasa (skala 0–2)

---

## Keterbatasan

1. **Label proxy.** OLX tidak menyediakan status terjual. Rujukan di atas mendukung arah hubungannya, tetapi label belum divalidasi terhadap penjualan nyata.
2. **Akurasi tinggi sebagian sirkular.** Tanpa `deviasi_persen`, akurasi hanya 51,40%.
3. **Transmisi 31–35% adalah asumsi** (iklan tidak menyebut transmisi, diisi `automatic`).
4. **Data jarang per model.** Sekitar 62–67% kombinasi merek+model+usia punya sampel kurang dari 15, sehingga banyak prediksi berkeyakinan rendah.
5. **Data snapshot.** Bukan real-time. Harga pasar dapat berubah.
6. **Validasi umur iklan tidak reliabel.** `tanggal_posting` kemungkinan tanggal dibuat, bukan diperbarui.
7. **Fokus penjual saja.** Tidak ada mode pembeli.

### Rencana validasi label

Cara terbaik memvalidasi label adalah memantau iklan yang sama dari waktu ke waktu dan mencatat kapan iklan menghilang, seperti pada Granát & Vékás (2025) dan Born dkk. (2018). Scraping ulang daftar iklan tidak cukup, karena hanya 185 iklan yang sama antara batch 1 dan batch 2. Yang diperlukan adalah mengunjungi ulang setiap `id_iklan` dan mencatat apakah halamannya masih aktif.

---

## Lisensi

Proyek akademik (Tugas Data Mining, TI24A2). Data bersumber dari scraping publik OLX.