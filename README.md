# AutoLaku

Sistem Prediksi Kecepatan Laku Mobil Bekas OLX Berdasarkan Harga Wajar, Jarak Tempuh, dan Deskripsi Iklan Menggunakan Metode Klasifikasi.

AutoLaku membantu **penjual** mobil bekas memperkirakan apakah rencana harga jualnya akan membuat mobilnya **Cepat**, **Sedang**, atau **Lambat** laku — dihitung dari data iklan mobil bekas OLX yang di-scraping dan dilatih dengan machine learning.

---

## Cara Kerja Singkat

1. Data iklan mobil bekas di-scraping dari OLX.
2. Data dibersihkan, model kendaraan diekstrak dari judul iklan, harga tidak wajar (boneka/placeholder) disaring.
3. Harga wajar tiap unit dihitung dengan model regresi (dengan fallback untuk kombinasi merek/model yang datanya sedikit).
4. Selisih (deviasi) antara harga wajar dan harga iklan dipakai untuk membentuk label **Cepat/Sedang/Lambat** (dibagi tertile — persentil 33 & 67 — bukan angka ambang bebas).
5. Sebuah classifier dilatih untuk memprediksi label ini dari data unit + deviasi harga.
6. Penjual memasukkan data mobilnya lewat aplikasi, sistem mengembalikan estimasi harga wajar dan prediksi kecepatan laku.

---

## Struktur File

| File | Fungsi |
|---|---|
| `scraping.py` | Mengambil data iklan mobil bekas dari OLX, disimpan ke `dataset_olx_mentah.csv`. |
| `kamus_slang_otomotif.py` | Kamus normalisasi slang & istilah jual-beli mobil (dipakai saat pembersihan teks). |
| `processing.py` | Membersihkan data mentah, ekstraksi model/merek/tahun/transmisi, deteksi garansi & indikasi kredit, hasil disimpan ke `dataset_olx_bersih.csv`. |
| `train_model_klasifikasi.py` | Melatih model regresi harga wajar & model classifier likuiditas, mengevaluasi, menyimpan `.pkl` + `autolaku_meta.json` + grafik evaluasi. |
| `mesin_prediksi.py` | Logika inti prediksi (validasi input, hitung harga wajar, panggil classifier) — dipakai bersama oleh CLI dan tes otomatis. |
| `test_prediksi_klasifikasi.py` | Antarmuka terminal untuk penjual: input data mobil, tampil hasil prediksi. |
| `test_sistem.py` | Tes otomatis (validasi input, deteksi negasi pada deskripsi, uji monotonisitas harga, dll). |

File yang dihasilkan otomatis (jangan diedit manual): `dataset_olx_bersih.csv`, `model_regresi_harga.pkl`, `model_klasifikasi_likuiditas.pkl`, `autolaku_meta.json`, folder `output_klasifikasi_<tanggal>/`.

---

## Cara Menjalankan

Pastikan semua file di atas ada dalam satu folder, beserta `dataset_olx_mentah.csv` (hasil scraping).

```bash
pip install pandas numpy scikit-learn matplotlib scipy

python processing.py                  # 1. Bersihkan data
python train_model_klasifikasi.py     # 2. Latih model
python test_sistem.py                 # 3. Jalankan tes otomatis (opsional tapi disarankan)
python test_prediksi_klasifikasi.py   # 4. Jalankan aplikasi prediksi
```

Urutan ini wajib — setiap langkah butuh hasil dari langkah sebelumnya. Kalau `dataset_olx_mentah.csv` diperbarui (scraping ulang) atau kode `processing.py`/`train_model_klasifikasi.py` diubah, ulangi dari langkah 1.

---

## Fitur yang Digunakan untuk Prediksi

- **Harga wajar** — dari merek, model, tahun, transmisi, jarak tempuh (deviasi harga jual terhadap ini adalah fitur utama).
- **Kilometer** — jarak tempuh total dan kilometer per tahun.
- **Deskripsi iklan** — dibaca untuk mendeteksi ada/tidaknya garansi dan indikasi kredit/TDP (termasuk mengenali negasi, mis. "tidak ada garansi").

Kolom lain (merek, model, tahun, transmisi) ikut menjadi fitur pendukung karena harga wajar sendiri bergantung padanya.

**Tidak dipakai sebagai fitur prediksi:** lokasi, tipe penjual (selalu "Individu" di aplikasi), foto/kondisi visual, umur iklan (umur iklan hanya dipakai sebagai validasi kasar di tahap training, lihat bagian Keterbatasan).

---

## Keterbatasan Sistem

1. **Label bersifat proxy, bukan status "terjual" sungguhan.** OLX tidak menyediakan data status terjual atau tanggal iklan turun, sehingga label Cepat/Sedang/Lambat dibentuk dari deviasi harga terhadap harga wajar, bukan observasi historis nyata.
2. **Validasi terhadap kenyataan masih lemah.** Umur iklan dipakai sebagai pendekatan validasi, tapi korelasinya kecil (rho ≈ 0.06) — jangan menyimpulkan sistem "terbukti akurat" dari sini.
3. **Akurasi tinggi classifier bersifat sebagian sirkular.** Fitur `deviasi_persen` dipakai membentuk label sekaligus jadi fitur prediksi. Uji ablation menunjukkan tanpa fitur ini akurasi turun tajam — akurasi tinggi sebagian besar mencerminkan classifier "menghafal" aturan tertile, bukan murni memprediksi realita pasar.
4. **Cakupan merek & model terbatas** pada daftar di `MODEL_PER_MEREK` (`processing.py`). Merek/model di luar daftar itu, atau yang field model-nya tidak diisi OLX, jatuh ke kategori "Lainnya".
5. **Kombinasi merek+model+usia dengan sampel sedikit** (di bawah ambang 15) ditandai "keyakinan rendah" pada hasil prediksi, dan harga wajarnya dihitung dengan pendekatan fallback (regresi usia), bukan model ML utama — akurasinya lebih rendah untuk mobil langka/tua.
6. **Data bersifat snapshot**, bukan real-time. Tidak ada pembaruan otomatis ketika iklan di OLX sudah dihapus atau berubah harga; perlu scraping ulang secara berkala.
7. **Hanya melayani sudut pandang penjual** (input mandiri spesifikasi & rencana harga jual), tidak ada mode untuk menjelajah iklan yang sudah ada di pasar.

---

## Metrik Evaluasi (terakhir dilatih: lihat `autolaku_meta.json`)

- **Model harga wajar:** MAPE, MAE, MSE, RMSE (grafik `7_mae_rmse_harga_wajar.png`).
- **Model classifier:** Accuracy, F1/Precision/Recall (macro), confusion matrix, RMSE ordinal (mengukur seberapa jauh kesalahan kelas, bukan cuma benar/salah), hasil uji ablation, dan akurasi dipecah per kecukupan sampel — semua di folder `output_klasifikasi_<tanggal>/` dan `ringkasan_metrik_klasifikasi.csv`.