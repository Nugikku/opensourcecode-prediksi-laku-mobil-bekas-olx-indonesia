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
| `tambah_data.py` | Menambah data latih baru, melatih ulang otomatis, mencatat riwayat sebelum vs sesudah (`riwayat_data.json`), dan menyediakan Guidebook PDF |
| `mesin_prediksi.py` | Engine prediksi (dipakai CLI, tes, FastAPI web, dan Streamlit) |
| `test_prediksi_klasifikasi.py` | Antarmuka terminal untuk penjual |
| `test_sistem.py` | 21 tes otomatis |
| `server.py` | Server web utama (FastAPI + Jinja2 + Tailwind CSS): Landing Page (`/`), Kalkulator (`/prediksi`), Tambah Data & Riwayat (`/tambah-data`), dan Unduh Guidebook (`/download/guidebook`) |
| `app.py` | Antarmuka alternatif berbasis Streamlit |

File yang dihasilkan otomatis: `dataset_olx_bersih.csv`, `dataset_olx_berlabel.csv`, `fitur_final.json`, `model_regresi_harga.pkl`, `model_klasifikasi_likuiditas.pkl`, `autolaku_meta.json`, `keputusan_fitur.csv`, `laporan_preprocessing.txt`, `riwayat_data.json`, `static/guidebook_autolaku.pdf`.

---

## Cara Menjalankan

```bash
pip install pandas numpy scikit-learn matplotlib streamlit fastapi uvicorn jinja2

python processing.py
python pelabelan.py
python analisis_fitur.py
python train_model_klasifikasi.py     # beberapa menit (nested CV)
python test_sistem.py                 # target: 21/21 lulus

python server.py                      # antarmuka web utama (http://localhost:8000)
streamlit run app.py                  # antarmuka Streamlit alternatif
python test_prediksi_klasifikasi.py   # antarmuka terminal
```

> Hapus `.pkl` dan `autolaku_meta.json` lama sebelum menjalankan ulang dari awal. Jangan lewati `analisis_fitur.py`, karena `train_model_klasifikasi.py` membaca `fitur_final.json`.
> Jika menggabungkan beberapa batch scraping, gunakan menu `/tambah-data` di web atau `python tambah_data.py`.

---

## Menambah Data Latih Baru & Audit Perubahan Model

Sistem dirancang untuk menjawab pertanyaan **"Bagaimana jika ada data baru?"** dengan fitur penambahan data dan pelatihan ulang otomatis (*Batch Retraining Pipeline*) yang dapat diakses langsung melalui website (`http://localhost:8000/tambah-data`) maupun terminal.

### 1. Melalui Menu Website (`/tambah-data`)
Buka `http://localhost:8000/tambah-data` setelah menjalankan `python server.py`. Tersedia tiga fasilitas utama:
- **Unduh Guidebook PDF (`/download/guidebook`):** Mengunduh buku panduan resmi cara melakukan scraping manual dari OLX serta panduan penggunaan aplikasi.
- **Opsi A (Unggah CSV Hasil Scraping Baru & Retraining):** Mengunggah file `.csv` hasil `scraping.py` untuk digabungkan ke dataset lama, dilatih ulang secara menyeluruh, dan menampilkan perbandingan lengkap *Sebelum vs Sesudah* di setiap tahap pipeline.
- **Opsi B (Percobaan Tambah Data Tanpa CSV):** Sesuai aturan integritas data sistem, percobaan menambah data tanpa CSV hasil scraping akan ditolak dengan notifikasi:
  > `"harus scraping manual, cara scraping dan penggunaan aplikasi ada di guidebook silahkan di download dan dibaca"`

### 2. Melalui Terminal (`tambah_data.py`)

```bash
python tambah_data.py hasil_scraping_baru.csv
python tambah_data.py hasil_scraping_baru.csv --tanpa-latih   # hanya gabung data
python tambah_data.py hasil_scraping_baru.csv --paksa         # tetap simpan walau tes gagal / 0 baris baru
```

### 3. Melalui Aplikasi Streamlit (`app.py`)
Buka sidebar `streamlit run app.py`, pilih **Admin: Tambah Data Latih & Retraining**, unggah CSV pada Opsi A, atau unduh Guidebook PDF.

### Tahapan yang Dijalankan Sistem Secara Otomatis:
1. Memeriksa kesesuaian 13 kolom wajib CSV baru dengan `dataset_olx_mentah.csv` dan membuang iklan duplikat yang `id_iklan`-nya sudah terdaftar.
2. Mengarsipkan seluruh dataset dan model lama ke folder `arsip/<timestamp>/`.
3. Menjalankan `processing.py` → `pelabelan.py` → `analisis_fitur.py` → `train_model_klasifikasi.py` → `test_sistem.py` (21 tes otomatis).
4. Jika ada tahap yang gagal, seluruh file dipulihkan otomatis (*auto-rollback*) dari folder arsip.
5. Mencatat riwayat perubahan lengkap (mulai dari jumlah data mentah/bersih, merek/model baru, pergeseran ambang kuantil `Q33`/`Q67`, MAPE/R² regresi, seleksi fitur, F1/akurasi klasifikasi, uji ablation, hingga 6 kandidat model) ke `riwayat_data.json` dan menampilkannya pada tabel perbandingan di web.

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
| Model terpilih | Random Forest | Random Forest |
| F1-macro CV (5-fold) | 99,41% ± 0,46% | 98,73% ± 0,64% |
| F1-macro test | 98,72% | 98,40% |
| Akurasi test | 98,71% | 98,49% |
| RMSE Ordinal test | 0,1136 | 0,1230 |
| Akurasi tanpa `deviasi_persen` (ablation) | 51,40% (F1 50,76%) | 42,91% (F1 41,87%) |

Kedua batch secara konsisten memilih **Random Forest** sebagai model klasifikasi terbaik dengan F1-macro di atas 98,4% pada data uji.

### Tes otomatis

| | Batch 1 | Batch 2 |
|---|---|---|
| Hasil | 21/21 lulus | 21/21 lulus |

Seluruh 21 tes integritas sistem pada `test_sistem.py` lulus 100% pada kedua batch setelah pengujian mengambil kasus secara dinamis dari dataset aktif.

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