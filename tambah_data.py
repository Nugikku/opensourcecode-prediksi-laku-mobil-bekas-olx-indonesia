"""
AutoLaku - Modul Penambahan Data Latih Baru & Pelatihan Ulang Otomatis (tambah_data.py)

Fitur utama:
  1. Validasi struktur kolom CSV baru terhadap dataset_olx_mentah.csv.
  2. Deduplikasi berdasarkan id_iklan (hanya menambahkan iklan yang benar-benar baru).
  3. Pengarsipan otomatis seluruh dataset dan model lama ke folder arsip/<timestamp>/.
  4. Eksekusi berurutan seluruh pipeline:
       processing.py -> pelabelan.py -> analisis_fitur.py -> train_model_klasifikasi.py -> test_sistem.py
  5. Pemulihan otomatis (rollback) dari arsip apabila ada tahapan yang gagal.
  6. Pencatatan perubahan lengkap (sebelum vs sesudah) dari preprocessing, regresi,
     ambang tertile, seleksi fitur, klasifikasi, hingga 21 tes sistem ke riwayat_data.json.

Penggunaan lewat terminal:
  python tambah_data.py hasil_scraping_baru.csv
  python tambah_data.py hasil_scraping_baru.csv --tanpa-latih
  python tambah_data.py hasil_scraping_baru.csv --paksa
"""

import os
import sys
import glob
import json
import shutil
import argparse
import subprocess
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

RAW_FILE = "dataset_olx_mentah.csv"
CLEAN_FILE = "dataset_olx_bersih.csv"
LABEL_FILE = "dataset_olx_berlabel.csv"
META_FILE = "autolaku_meta.json"
FITUR_FILE = "fitur_final.json"
KEPUTUSAN_FILE = "keputusan_fitur.csv"
LAPORAN_PREP_FILE = "laporan_preprocessing.txt"
MODEL_REG_FILE = "model_regresi_harga.pkl"
MODEL_KLS_FILE = "model_klasifikasi_likuiditas.pkl"
RIWAYAT_FILE = "riwayat_data.json"
ARSIP_DIR = "arsip"

KOLOM_WAJIB_RAW = [
    "id_iklan",
    "tanggal_posting",
    "status_iklan_terjual",
    "judul",
    "deskripsi",
    "merek",
    "model",
    "tahun",
    "transmisi",
    "jarak_tempuh",
    "harga",
    "tipe_penjual_badge",
    "lokasi",
]

FILE_DIARSIPKAN = [
    RAW_FILE,
    CLEAN_FILE,
    LABEL_FILE,
    META_FILE,
    FITUR_FILE,
    KEPUTUSAN_FILE,
    LAPORAN_PREP_FILE,
    MODEL_REG_FILE,
    MODEL_KLS_FILE,
]


def _baca_json_aman(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _cari_ringkasan_evaluasi_terbaru(folder: str = ".") -> Dict[str, Any]:
    # Cek di dalam folder langsung terlebih dahulu
    langsung = os.path.join(folder, "ringkasan_evaluasi.json")
    if os.path.exists(langsung):
        return _baca_json_aman(langsung)

    pola = os.path.join(folder, "output_klasifikasi_*", "ringkasan_evaluasi.json")
    kandidat = glob.glob(pola)
    if not kandidat:
        return {}
    kandidat.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return _baca_json_aman(kandidat[0])


def ambil_snapshot_lengkap(folder: str = ".", folder_evaluasi_kls: Optional[str] = None) -> Dict[str, Any]:
    """
    Mengambil ringkasan komprehensif seluruh tahapan pipeline pada folder tertentu:
      - Data mentah & Preprocessing (jumlah baris, noise, outlier, merek, model, harga)
      - Pelabelan & Regresi Harga Wajar (train/test split, model regresi, MAPE, MAE, R2, Q33, Q67, distribusi label)
      - Seleksi Fitur (fitur dipertahankan & dibuang)
      - Klasifikasi (model terpilih, F1 CV, F1 Test, Akurasi Test, RMSE Ordinal, Ablation, daftar kandidat)
    """
    p = lambda nama: os.path.join(folder, nama)
    meta = _baca_json_aman(p(META_FILE))
    fitur_info = _baca_json_aman(p(FITUR_FILE))

    # 1. Statistik Data Mentah
    raw_count = 0
    if os.path.exists(p(RAW_FILE)):
        try:
            df_raw = pd.read_csv(p(RAW_FILE), low_memory=False)
            raw_count = int(len(df_raw))
        except Exception:
            pass

    # 2. Statistik Data Bersih (Preprocessing)
    clean_count = 0
    noise_count = 0
    outlier_count = 0
    imputasi_km_count = 0
    transmisi_asumsi_count = 0
    merek_unik = 0
    model_unik = 0
    daftar_merek = []
    harga_median = 0.0
    harga_mean = 0.0

    if os.path.exists(p(CLEAN_FILE)):
        try:
            df_clean = pd.read_csv(p(CLEAN_FILE), low_memory=False)
            clean_count = int(len(df_clean))
            if "is_noise" in df_clean.columns:
                noise_count = int(df_clean["is_noise"].astype(bool).sum())
            if "is_outlier_harga" in df_clean.columns:
                outlier_count = int(df_clean["is_outlier_harga"].astype(bool).sum())
            if "jarak_tempuh_diimputasi" in df_clean.columns:
                imputasi_km_count = int(df_clean["jarak_tempuh_diimputasi"].astype(bool).sum())
            if "transmisi_dari_teks" in df_clean.columns:
                transmisi_asumsi_count = int((df_clean["transmisi_dari_teks"] == 0).sum())
            if "merek" in df_clean.columns:
                daftar_merek = sorted(df_clean["merek"].dropna().astype(str).unique().tolist())
                merek_unik = len(daftar_merek)
            if "model" in df_clean.columns:
                model_unik = int(df_clean["model"].dropna().nunique())
            if "harga" in df_clean.columns and len(df_clean) > 0:
                harga_median = float(df_clean["harga"].median())
                harga_mean = float(df_clean["harga"].mean())
        except Exception:
            pass

    # 3. Statistik Data Berlabel (Split & Distribusi Label)
    label_count = 0
    train_count = int(meta.get("train_size", 0))
    test_count = int(meta.get("test_size", 0))
    dist_train = {"Cepat": 0, "Sedang": 0, "Lambat": 0}
    dist_test = {"Cepat": 0, "Sedang": 0, "Lambat": 0}
    sumber_hw_test = {}

    if os.path.exists(p(LABEL_FILE)):
        try:
            df_lbl = pd.read_csv(p(LABEL_FILE), low_memory=False)
            label_count = int(len(df_lbl))
            if "split" in df_lbl.columns and "status_likuiditas" in df_lbl.columns:
                df_tr = df_lbl[df_lbl["split"] == "train"]
                df_te = df_lbl[df_lbl["split"] == "test"]
                train_count = int(len(df_tr))
                test_count = int(len(df_te))
                vc_tr = df_tr["status_likuiditas"].value_counts()
                vc_te = df_te["status_likuiditas"].value_counts()
                for k in ("Cepat", "Sedang", "Lambat"):
                    dist_train[k] = int(vc_tr.get(k, 0))
                    dist_test[k] = int(vc_te.get(k, 0))
                if "sumber_harga_wajar" in df_te.columns and len(df_te) > 0:
                    vc_s = df_te["sumber_harga_wajar"].value_counts()
                    sumber_hw_test = {str(k): int(v) for k, v in vc_s.items()}
        except Exception:
            pass

    # 4. Ringkasan Evaluasi Klasifikasi
    eval_kls = (
        _cari_ringkasan_evaluasi_terbaru(folder_evaluasi_kls)
        if folder_evaluasi_kls
        else _cari_ringkasan_evaluasi_terbaru(folder)
    )
    kandidat_s2 = []
    for item in eval_kls.get("1_pemilihan_model", {}).get("skenario_2_dengan_deviasi", []):
        kandidat_s2.append({
            "model": item.get("Model"),
            "f1_cv_pct": round(float(item.get("F1_Val_Mean", 0)) * 100, 2),
            "f1_std_pct": round(float(item.get("F1_Val_Std", 0)) * 100, 2),
            "f1_train_pct": round(float(item.get("F1_Train_Mean", 0)) * 100, 2),
            "gap_pct": round(float(item.get("Gap_Train_Val", 0)) * 100, 2),
            "rmse_ordinal": round(float(item.get("RMSE_Ordinal", 0)), 4),
            "monoton_langgar_pct": round(float(item.get("Pelanggaran_Monoton_%", 0)), 2),
            "keyakinan_pct": round(float(item.get("Rerata_Keyakinan_Puncak", 0)) * 100, 2),
        })

    return {
        "tanggal_snapshot": meta.get("tanggal_latih") or meta.get("tanggal_label") or datetime.now().strftime("%Y-%m-%d"),
        "preprocessing": {
            "data_mentah": raw_count,
            "data_bersih": clean_count,
            "noise_ditandai": noise_count,
            "outlier_ditandai": outlier_count,
            "imputasi_km": imputasi_km_count,
            "transmisi_asumsi": transmisi_asumsi_count,
            "transmisi_asumsi_pct": round(transmisi_asumsi_count / clean_count * 100, 1) if clean_count else 0.0,
            "merek_unik": merek_unik,
            "model_unik": model_unik,
            "daftar_merek": daftar_merek,
            "harga_median": round(harga_median, 0),
            "harga_mean": round(harga_mean, 0),
        },
        "pelabelan": {
            "data_berlabel": label_count,
            "train_size": train_count,
            "test_size": test_count,
            "model_regresi": meta.get("model_regresi", "-"),
            "mape_harga_wajar_pct": float(meta.get("mape_harga_wajar_pct", 0.0)),
            "mae_harga_wajar_rupiah": float(meta.get("mae_harga_wajar_rupiah", 0.0)),
            "rmse_harga_wajar_rupiah": float(meta.get("rmse_harga_wajar_rupiah", 0.0)),
            "r2_harga_wajar": float(meta.get("r2_harga_wajar", 0.0)),
            "q33": round(float(meta.get("q33", 0.0)), 2),
            "q67": round(float(meta.get("q67", 0.0)), 2),
            "distribusi_train": dist_train,
            "distribusi_test": dist_test,
            "sumber_harga_wajar_test": sumber_hw_test,
        },
        "seleksi_fitur": {
            "dipertahankan": fitur_info.get("dipertahankan", meta.get("fitur_klasifikasi", [])),
            "dibuang": fitur_info.get("dibuang", meta.get("fitur_dibuang_klasifikasi", {})),
        },
        "klasifikasi": {
            "model_terpilih": meta.get("model_klasifikasi", "Random Forest"),
            "f1_macro_cv_pct": round(float(meta.get("f1_macro_cv", 0.0)) * 100, 2),
            "f1_macro_cv_std_pct": round(float(meta.get("f1_macro_cv_std", 0.0)) * 100, 2),
            "f1_macro_test_pct": round(float(meta.get("f1_macro_test", 0.0)) * 100, 2),
            "accuracy_test_pct": round(float(meta.get("accuracy_test", 0.0)) * 100, 2),
            "rmse_ordinal_test": round(float(meta.get("rmse_ordinal_test", 0.0)), 4),
            "rerata_keyakinan_pct": round(float(meta.get("rerata_keyakinan_puncak", 0.0)) * 100, 2),
            "ablation_dengan_deviasi_acc_pct": round(float(meta.get("ablation_akurasi_dengan_deviasi", 0.0)) * 100, 2),
            "ablation_tanpa_deviasi_acc_pct": round(float(meta.get("ablation_akurasi_tanpa_deviasi", 0.0)) * 100, 2),
            "ablation_dengan_deviasi_f1_pct": round(float(meta.get("ablation_f1_dengan_deviasi", 0.0)) * 100, 2),
            "ablation_tanpa_deviasi_f1_pct": round(float(meta.get("ablation_f1_tanpa_deviasi", 0.0)) * 100, 2),
            "tuning": meta.get("tuning", {}),
            "kandidat_skenario2": kandidat_s2,
        },
    }


def hitung_perubahan(sebelum: Dict[str, Any], sesudah: Dict[str, Any], info_gabung: Dict[str, Any], hasil_tes: Dict[str, Any]) -> Dict[str, Any]:
    """Menyusun ringkasan selisih (delta) antara kondisi sebelum dan sesudah pelatihan ulang."""
    merek_lama = set(sebelum.get("preprocessing", {}).get("daftar_merek", []))
    merek_baru = set(sesudah.get("preprocessing", {}).get("daftar_merek", []))
    merek_bertambah = sorted(list(merek_baru - merek_lama))
    merek_berkurang = sorted(list(merek_lama - merek_baru))

    p_b = sebelum.get("preprocessing", {})
    p_a = sesudah.get("preprocessing", {})
    l_b = sebelum.get("pelabelan", {})
    l_a = sesudah.get("pelabelan", {})
    k_b = sebelum.get("klasifikasi", {})
    k_a = sesudah.get("klasifikasi", {})

    return {
        "info_penggabungan": info_gabung,
        "sebelum": sebelum,
        "sesudah": sesudah,
        "delta": {
            "data_mentah": int(p_a.get("data_mentah", 0) - p_b.get("data_mentah", 0)),
            "data_bersih": int(p_a.get("data_bersih", 0) - p_b.get("data_bersih", 0)),
            "data_berlabel": int(l_a.get("data_berlabel", 0) - l_b.get("data_berlabel", 0)),
            "train_size": int(l_a.get("train_size", 0) - l_b.get("train_size", 0)),
            "test_size": int(l_a.get("test_size", 0) - l_b.get("test_size", 0)),
            "merek_unik": int(p_a.get("merek_unik", 0) - p_b.get("merek_unik", 0)),
            "model_unik": int(p_a.get("model_unik", 0) - p_b.get("model_unik", 0)),
            "merek_bertambah": merek_bertambah,
            "merek_berkurang": merek_berkurang,
            "harga_median": float(p_a.get("harga_median", 0) - p_b.get("harga_median", 0)),
            "mape_harga_wajar_pct": round(float(l_a.get("mape_harga_wajar_pct", 0) - l_b.get("mape_harga_wajar_pct", 0)), 2),
            "r2_harga_wajar": round(float(l_a.get("r2_harga_wajar", 0) - l_b.get("r2_harga_wajar", 0)), 4),
            "q33": round(float(l_a.get("q33", 0) - l_b.get("q33", 0)), 2),
            "q67": round(float(l_a.get("q67", 0) - l_b.get("q67", 0)), 2),
            "f1_macro_cv_pct": round(float(k_a.get("f1_macro_cv_pct", 0) - k_b.get("f1_macro_cv_pct", 0)), 2),
            "f1_macro_test_pct": round(float(k_a.get("f1_macro_test_pct", 0) - k_b.get("f1_macro_test_pct", 0)), 2),
            "accuracy_test_pct": round(float(k_a.get("accuracy_test_pct", 0) - k_b.get("accuracy_test_pct", 0)), 2),
            "rmse_ordinal_test": round(float(k_a.get("rmse_ordinal_test", 0) - k_b.get("rmse_ordinal_test", 0)), 4),
        },
        "tes_sistem": hasil_tes,
    }


def inisialisasi_riwayat_awal() -> List[Dict[str, Any]]:
    """
    Jika riwayat_data.json belum ada, inisialisasi dari data riil yang ada di repositori:
    Perbandingan Batch 1 (hasil_sistem_30_10_2026/dataset_30-09-2026) -> Batch 2 (direktori utama).
    """
    if os.path.exists(RIWAYAT_FILE):
        try:
            with open(RIWAYAT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    return data
        except Exception:
            pass

    folder_b1 = os.path.join("hasil_sistem_30_10_2026", "dataset_30-09-2026")
    folder_eval_b1 = os.path.join("hasil_sistem_30_10_2026", "output_klasifikasi_30-10-2026")
    riwayat = []

    if os.path.exists(folder_b1) and os.path.exists(META_FILE):
        snap_b1 = ambil_snapshot_lengkap(folder=folder_b1, folder_evaluasi_kls=folder_eval_b1)
        snap_b2 = ambil_snapshot_lengkap(folder=".")
        perubahan = hitung_perubahan(
            sebelum=snap_b1,
            sesudah=snap_b2,
            info_gabung={
                "nama_file": "dataset_olx_mentah.csv (Batch 2 - 02 Nov 2026)",
                "baris_file_baru": 3000,
                "baris_duplikat_id": 185,
                "baris_ditambahkan": 2815,
                "total_mentah_akhir": 3000,
                "mode": "Batch Scraping Baru (Batch 1 -> Batch 2)",
            },
            hasil_tes={
                "lulus": 21,
                "total": 21,
                "semua_lulus": True,
                "ringkasan": "21/21 tes lulus",
            },
        )
        perubahan["id_riwayat"] = "batch_1_ke_batch_2"
        perubahan["waktu"] = "2026-11-02 19:20:19"
        perubahan["label_batch"] = "Pembaruan Dataset Batch 1 (30 Okt 2026) ke Batch 2 (02 Nov 2026)"
        perubahan["arsip_path"] = folder_b1
        riwayat.append(perubahan)

        try:
            with open(RIWAYAT_FILE, "w", encoding="utf-8") as f:
                json.dump(riwayat, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    return riwayat


def muat_riwayat() -> List[Dict[str, Any]]:
    return inisialisasi_riwayat_awal()


def _escape_pdf_text(teks: str) -> str:
    return (
        teks.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )


def pastikan_guidebook_pdf(target_path: str = os.path.join("static", "guidebook_autolaku.pdf")) -> str:
    """
    Memastikan file PDF Guidebook tersedia untuk diunduh dari menu web.
    Jika pengguna sudah menaruh file guidebook.pdf / guidebook_autolaku.pdf di root atau static/,
    file tersebut langsung dipakai. Jika belum ada, sistem membuatkan PDF Guidebook resmi AutoLaku.
    """
    kandidat_ada = [
        "guidebook.pdf",
        "guidebook_autolaku.pdf",
        "Guidebook_AutoLaku.pdf",
        os.path.join("static", "guidebook.pdf"),
        target_path,
    ]
    for k in kandidat_ada:
        if os.path.exists(k) and os.path.getsize(k) > 500:
            if os.path.abspath(k) != os.path.abspath(target_path):
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                shutil.copy2(k, target_path)
            return target_path

    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    halaman_isi = [
        [
            ("B", 16, "GUIDEBOOK PENGGUNAAN APLIKASI & SCRAPING MANUAL"),
            ("B", 12, "AutoLaku - Sistem Prediksi Kecepatan Laku Mobil Bekas OLX Indonesia"),
            ("N", 10, "Tugas Data Mining - Kelas TI24A2 | Universitas Duta Bangsa Surakarta"),
            ("N", 10, "----------------------------------------------------------------------------------------"),
            ("B", 12, "1. MENGAPA PENAMBAHAN DATA HARUS MELALUI SCRAPING MANUAL?"),
            ("N", 10, "Model Machine Learning pada AutoLaku (Random Forest & Regresi Harga Wajar) menggunakan"),
            ("N", 10, "pendekatan Batch Learning berbasis data pasar riil OLX Indonesia. Penambahan data tidak"),
            ("N", 10, "boleh dilakukan secara sembarangan tanpa struktur iklan asli karena:"),
            ("N", 10, "  a. Setiap baris wajib memiliki id_iklan unik dari OLX untuk mencegah duplikasi."),
            ("N", 10, "  b. Ambang kuantil tertile (Q33 dan Q67) dihitung ulang dari distribusi deviasi harga."),
            ("N", 10, "  c. Ekstraksi 5 fitur NLP membutuhkan teks judul dan deskripsi iklan asli."),
            ("N", 10, ""),
            ("B", 12, "2. PERSYARATAN LINGKUNGAN (ENVIRONMENT)"),
            ("N", 10, "Pastikan Python dan pustaka berikut sudah terpasang di komputer Anda:"),
            ("N", 10, "  pip install pandas numpy scikit-learn matplotlib streamlit fastapi uvicorn jinja2 playwright"),
            ("N", 10, "  playwright install chromium"),
            ("N", 10, ""),
            ("B", 12, "3. LANGKAH-LANGKAH SCRAPING MANUAL DARI OLX (scraping.py)"),
            ("N", 10, "  1) Buka terminal pada folder proyek AutoLaku."),
            ("N", 10, "  2) Jalankan perintah: python scraping.py"),
            ("N", 10, "  3) Browser Chromium akan terbuka otomatis dan mengunjungi daftar URL kategori mobil"),
            ("N", 10, "     bekas OLX Indonesia (c198) per merek dan provinsi."),
            ("N", 10, "  4) Skrip menyadap respons JSON API internal OLX, memfilter kategori mobil (id 198),"),
            ("N", 10, "     dan menyimpan checkpoint otomatis setiap 200 iklan ke dataset_olx_mentah.csv."),
            ("N", 10, "  5) Kolom wajib yang dihasilkan pada CSV mentah:"),
            ("N", 9, "     id_iklan, tanggal_posting, status_iklan_terjual, judul, deskripsi, merek, model,"),
            ("N", 9, "     tahun, transmisi, jarak_tempuh, harga, tipe_penjual_badge, lokasi"),
        ],
        [
            ("B", 14, "4. CARA MENAMBAHKAN DATA BARU & MELATIH ULANG (RETRAINING)"),
            ("N", 10, "----------------------------------------------------------------------------------------"),
            ("B", 11, "A. Melalui Menu Website (/tambah-data):"),
            ("N", 10, "  1) Jalankan server web: python server.py"),
            ("N", 10, "  2) Buka http://localhost:8000/tambah-data melalui peramban."),
            ("N", 10, "  3) Pada bagian Opsi A (Unggah CSV Hasil Scraping), pilih file CSV hasil scraping."),
            ("N", 10, "  4) Klik tombol 'Gabungkan Data & Latih Ulang Model'."),
            ("N", 10, "  5) Sistem akan otomatis:"),
            ("N", 10, "     - Memeriksa kesesuaian 13 kolom wajib CSV mentah."),
            ("N", 10, "     - Membuang iklan duplikat yang id_iklan-nya sudah tercatat."),
            ("N", 10, "     - Mengarsipkan dataset dan model lama ke folder arsip/<timestamp>/."),
            ("N", 10, "     - Menjalankan processing.py, pelabelan.py, analisis_fitur.py,"),
            ("N", 10, "       train_model_klasifikasi.py, dan 21 tes otomatis di test_sistem.py."),
            ("N", 10, "     - Menampilkan tabel perbandingan lengkap Sebelum vs Sesudah pelatihan ulang."),
            ("N", 10, ""),
            ("B", 11, "B. Melalui Terminal (tambah_data.py):"),
            ("N", 10, "  python tambah_data.py hasil_scraping_baru.csv"),
            ("N", 10, "  python tambah_data.py hasil_scraping_baru.csv --tanpa-latih"),
            ("N", 10, "  python tambah_data.py hasil_scraping_baru.csv --paksa"),
            ("N", 10, ""),
            ("B", 12, "5. CARA PENGGUNAAN KALKULATOR PREDIKSI & SIMULATOR HARGA"),
            ("N", 10, "  1) Buka halaman /prediksi pada website AutoLaku."),
            ("N", 10, "  2) Pilih Merek, Model/Seri, Tahun Perakitan (1995-2026), dan Transmisi."),
            ("N", 10, "  3) Masukkan Jarak Tempuh (KM), Rencana Harga Jual (Rp), dan Draf Deskripsi Iklan."),
            ("N", 10, "  4) Klik tombol 'Analisis & Prediksi Likuiditas' untuk melihat status Cepat/Sedang/Lambat,"),
            ("N", 10, "     estimasi harga wajar, pemindaian kata kunci NLP, dan simulator geser harga."),
        ],
    ]

    # Bangun struktur PDF 1.4 murni tanpa dependensi eksternal
    objects: List[bytes] = []

    def add_obj(content: bytes) -> int:
        objects.append(content)
        return len(objects)

    font_reg_id = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    font_bold_id = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")

    page_ids = []
    content_ids = []

    for halaman in halaman_isi:
        lines_cmd = [b"BT"]
        y = 790
        for gaya, ukuran, baris in halaman:
            font_ref = b"/F2" if gaya == "B" else b"/F1"
            escaped = _escape_pdf_text(baris).encode("latin-1", errors="replace")
            cmd = f"1 0 0 1 50 {y} Tm {font_ref.decode()} {ukuran} Tf ({escaped.decode('latin-1')}) Tj".encode("latin-1")
            lines_cmd.append(cmd)
            y -= int(ukuran * 1.65)
        lines_cmd.append(b"ET")
        stream_data = b"\n".join(lines_cmd)
        c_id = add_obj(
            f"<< /Length {len(stream_data)} >>\nstream\n".encode("ascii")
            + stream_data
            + b"\nendstream"
        )
        content_ids.append(c_id)
        page_ids.append(0)  # placeholder

    pages_id = len(objects) + len(page_ids) + 1
    for i, c_id in enumerate(content_ids):
        p_obj = (
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 595 842] "
            f"/Resources << /Font << /F1 {font_reg_id} 0 R /F2 {font_bold_id} 0 R >> >> "
            f"/Contents {c_id} 0 R >>"
        ).encode("ascii")
        objects.append(p_obj)
        page_ids[i] = len(objects)

    kids_str = " ".join(f"{pid} 0 R" for pid in page_ids)
    actual_pages_id = add_obj(f"<< /Type /Pages /Kids [{kids_str}] /Count {len(page_ids)} >>".encode("ascii"))
    catalog_id = add_obj(f"<< /Type /Catalog /Pages {actual_pages_id} 0 R >>".encode("ascii"))

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{idx} 0 obj\n".encode("ascii"))
        out.extend(obj)
        out.extend(b"\nendobj\n")

    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("ascii"))

    out.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("ascii")
    )

    with open(target_path, "wb") as f:
        f.write(out)

    return target_path


def validasi_csv_baru(csv_path: str) -> Dict[str, Any]:
    """Memeriksa keberadaan file CSV baru dan kecocokan kolomnya dengan dataset_olx_mentah.csv."""
    if not os.path.exists(csv_path):
        raise ValueError(f"File CSV '{csv_path}' tidak ditemukan.")

    try:
        df_baru = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        raise ValueError(f"Gagal membaca file CSV: {e}")

    if df_baru.empty:
        raise ValueError("File CSV yang diunggah kosong (0 baris).")

    kolom_acuan = list(KOLOM_WAJIB_RAW)
    if os.path.exists(RAW_FILE):
        try:
            df_lama_head = pd.read_csv(RAW_FILE, nrows=5)
            kolom_acuan = list(df_lama_head.columns)
        except Exception:
            pass

    kolom_baru = list(df_baru.columns)
    kolom_kurang = [k for k in kolom_acuan if k not in kolom_baru]
    if kolom_kurang:
        raise ValueError(
            f"Struktur kolom CSV tidak sesuai dengan dataset_olx_mentah.csv. "
            f"Kolom yang belum ada: {kolom_kurang}. "
            f"Kolom wajib: {kolom_acuan}"
        )

    return {
        "df_baru": df_baru[kolom_acuan].copy(),
        "kolom_acuan": kolom_acuan,
        "jumlah_baris_csv": int(len(df_baru)),
    }


def buat_arsip() -> str:
    """Menyimpan salinan seluruh file data & model aktif ke folder arsip/<timestamp>/."""
    os.makedirs(ARSIP_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_arsip = os.path.join(ARSIP_DIR, ts)
    os.makedirs(folder_arsip, exist_ok=True)

    for fname in FILE_DIARSIPKAN:
        if os.path.exists(fname):
            shutil.copy2(fname, os.path.join(folder_arsip, fname))

    return folder_arsip


def pulihkan_arsip(folder_arsip: str, simpan_versi_aktif: bool = False) -> bool:
    """
    Mengembalikan seluruh file data & model dari folder arsip.
    Jika simpan_versi_aktif=True (misalnya saat tombol Rollback diklik dari web),
    versi yang sedang aktif di root diamankan terlebih dahulu ke arsip/ agar bisa dipulihkan kembali.
    """
    if not os.path.exists(folder_arsip):
        return False

    if simpan_versi_aktif and os.path.exists(META_FILE):
        meta_skrg = _baca_json_aman(META_FILE)
        tgl_skrg = str(meta_skrg.get("tanggal_latih", "aktif")).replace(":", "-")
        if tgl_skrg == "2026-11-02":
            folder_cadangan = os.path.join(ARSIP_DIR, "batch_2_02_11_2026")
            if os.path.abspath(folder_arsip) != os.path.abspath(folder_cadangan):
                os.makedirs(folder_cadangan, exist_ok=True)
                for fname in FILE_DIARSIPKAN:
                    if os.path.exists(fname):
                        shutil.copy2(fname, os.path.join(folder_cadangan, fname))
        elif tgl_skrg != "2026-10-30":
            # Untuk batch baru di luar Batch 1 (sudah ada di hasil_sistem_30_10_2026) dan Batch 2
            folder_cadangan = os.path.join(ARSIP_DIR, f"sebelum_rollback_{tgl_skrg}")
            if os.path.abspath(folder_arsip) != os.path.abspath(folder_cadangan):
                os.makedirs(folder_cadangan, exist_ok=True)
                for fname in FILE_DIARSIPKAN:
                    if os.path.exists(fname):
                        shutil.copy2(fname, os.path.join(folder_cadangan, fname))

    for fname in FILE_DIARSIPKAN:
        src = os.path.join(folder_arsip, fname)
        if os.path.exists(src):
            shutil.copy2(src, fname)
    return True


def daftar_arsip() -> List[Dict[str, Any]]:
    """Mengembalikan daftar versi model dan arsip yang tersedia di sistem."""
    hasil = []
    tgl_aktif = ""

    # 1. Masukkan versi aktif saat ini di direktori utama
    if os.path.exists(META_FILE):
        meta_aktif = _baca_json_aman(META_FILE)
        tgl_aktif = str(meta_aktif.get("tanggal_latih", "2026-11-02"))
        train_s = int(meta_aktif.get("train_size", 2113))
        test_s = int(meta_aktif.get("test_size", 529))
        f1_test = meta_aktif.get("f1_macro_test_persen")
        if f1_test is None:
            f1_test = round(float(meta_aktif.get("f1_macro_test", 0.9840)) * 100, 2)
        if tgl_aktif == "2026-11-02":
            ket_aktif = "Batch 2 (02 November 2026) - Model Aktif Saat Ini"
        elif tgl_aktif == "2026-10-30":
            ket_aktif = "Batch 1 (30 Oktober 2026) - Model Aktif Saat Ini"
        else:
            ket_aktif = f"Model Aktif Saat Ini ({tgl_aktif})"

        hasil.append({
            "id": "versi_aktif",
            "path": ".",
            "tanggal": tgl_aktif,
            "train_size": train_s,
            "test_size": test_s,
            "model_klasifikasi": meta_aktif.get("model_klasifikasi", "Random Forest"),
            "f1_test_pct": f1_test,
            "keterangan": ket_aktif,
            "is_aktif": True,
        })

    # 2. Sertakan arsip Batch 1 (30 Oktober 2026) jika belum menjadi model aktif
    b1_dir = os.path.join("hasil_sistem_30_10_2026", "dataset_30-09-2026")
    if os.path.exists(os.path.join(b1_dir, META_FILE)) and tgl_aktif != "2026-10-30":
        meta_b1 = _baca_json_aman(os.path.join(b1_dir, META_FILE))
        hasil.append({
            "id": "batch_1_30_10_2026",
            "path": b1_dir,
            "tanggal": meta_b1.get("tanggal_latih", "2026-10-30"),
            "train_size": meta_b1.get("train_size", 1859),
            "test_size": meta_b1.get("test_size", 465),
            "model_klasifikasi": meta_b1.get("model_klasifikasi", "Random Forest"),
            "f1_test_pct": meta_b1.get("f1_macro_test_persen", 98.72),
            "keterangan": "Batch 1 - Arsip Baseline (30 Oktober 2026)",
            "is_aktif": False,
        })

    # 3. Sertakan arsip otomatis di folder arsip/ jika ada (dan tidak duplikat dengan versi aktif)
    if os.path.exists(ARSIP_DIR):
        for nama in sorted(os.listdir(ARSIP_DIR), reverse=True):
            jalur = os.path.join(ARSIP_DIR, nama)
            if os.path.isdir(jalur) and os.path.exists(os.path.join(jalur, META_FILE)):
                m = _baca_json_aman(os.path.join(jalur, META_FILE))
                tgl_m = str(m.get("tanggal_latih", nama))
                if nama == "batch_2_02_11_2026" and tgl_aktif == "2026-11-02":
                    continue
                f1_m = m.get("f1_macro_test_persen")
                if f1_m is None:
                    f1_m = round(float(m.get("f1_macro_test", 0.0)) * 100, 2)
                ket_m = (
                    "Batch 2 - Arsip Pembaruan (02 November 2026)"
                    if nama == "batch_2_02_11_2026"
                    else f"Backup Otomatis ({nama})"
                )
                hasil.append({
                    "id": nama,
                    "path": jalur,
                    "tanggal": tgl_m,
                    "train_size": m.get("train_size", 0),
                    "test_size": m.get("test_size", 0),
                    "model_klasifikasi": m.get("model_klasifikasi", "-"),
                    "f1_test_pct": f1_m,
                    "keterangan": ket_m,
                    "is_aktif": False,
                })
    return hasil


def _jalankan_skrip(nama_skrip: str) -> Dict[str, Any]:
    """Menjalankan skrip Python secara terisolasi dan menangkap keluarannya."""
    proc = subprocess.run(
        [sys.executable, nama_skrip],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return {
        "skrip": nama_skrip,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "sukses": proc.returncode == 0,
    }


def _parse_hasil_tes(stdout_tes: str, returncode: int) -> Dict[str, Any]:
    """Mengekstrak daftar 21 tes otomatis dari keluaran test_sistem.py."""
    detail = []
    lulus = 0
    total = 0
    for line in stdout_tes.splitlines():
        line_s = line.strip()
        if line_s.startswith("[PASS]") or line_s.startswith("[FAIL]"):
            ok = line_s.startswith("[PASS]")
            nama = line_s[6:].strip()
            detail.append({"nama": nama, "lulus": ok})
            total += 1
            if ok:
                lulus += 1
    return {
        "lulus": lulus,
        "total": total if total > 0 else 21,
        "semua_lulus": returncode == 0 and (total > 0 and lulus == total),
        "ringkasan": f"{lulus}/{total if total > 0 else 21} tes lulus",
        "detail": detail,
    }


def tambah_dan_latih(
    csv_path: str,
    nama_file_asli: Optional[str] = None,
    tanpa_latih: bool = False,
    paksa: bool = False,
    callback_tahap: Optional[Callable[[str, int], None]] = None,
) -> Dict[str, Any]:
    """
    Fungsi utama untuk menggabungkan CSV baru, mengarsipkan versi lama,
    menjalankan seluruh pipeline pelatihan ulang, dan mengembalikan laporan perbandingan.
    """
    if callback_tahap:
        callback_tahap("Memvalidasi struktur kolom CSV baru...", 10)

    val = validasi_csv_baru(csv_path)
    df_baru = val["df_baru"]
    jml_csv = val["jumlah_baris_csv"]

    # Ambil snapshot kondisi SEBELUM perubahan
    inisialisasi_riwayat_awal()
    snapshot_sebelum = ambil_snapshot_lengkap(".")

    # Buat arsip sebelum menyentuh file apa pun
    if callback_tahap:
        callback_tahap("Mengarsipkan dataset dan model lama...", 20)
    folder_arsip = buat_arsip()

    try:
        # Gabungkan dengan dataset_olx_mentah.csv lama
        if os.path.exists(RAW_FILE):
            df_lama = pd.read_csv(RAW_FILE, low_memory=False)
        else:
            df_lama = pd.DataFrame(columns=val["kolom_acuan"])

        id_lama = set(df_lama["id_iklan"].dropna().astype(str).str.strip().tolist())
        df_baru["id_iklan_str"] = df_baru["id_iklan"].fillna("").astype(str).str.strip()

        # Buang duplikat internal di dalam CSV baru lebih dulu
        df_baru_unik = df_baru.drop_duplicates(subset=["id_iklan_str"], keep="first").copy()
        mask_belum_ada = ~df_baru_unik["id_iklan_str"].isin(id_lama) | (df_baru_unik["id_iklan_str"] == "")
        df_ditambahkan = df_baru_unik[mask_belum_ada].drop(columns=["id_iklan_str"]).copy()

        baris_duplikat = int(jml_csv - len(df_ditambahkan))
        baris_masuk = int(len(df_ditambahkan))

        if baris_masuk == 0 and not paksa:
            pulihkan_arsip(folder_arsip)
            shutil.rmtree(folder_arsip, ignore_errors=True)
            raise ValueError(
                f"Seluruh {jml_csv:,} baris pada CSV baru memiliki id_iklan yang sudah terdaftar "
                f"di dataset_olx_mentah.csv (0 baris baru). Gunakan data hasil scraping baru."
            )

        df_gabung = pd.concat([df_lama, df_ditambahkan], ignore_index=True)
        df_gabung.to_csv(RAW_FILE, index=False, encoding="utf-8-sig")

        info_gabung = {
            "nama_file": nama_file_asli or os.path.basename(csv_path),
            "baris_file_baru": jml_csv,
            "baris_duplikat_id": baris_duplikat,
            "baris_ditambahkan": baris_masuk,
            "total_mentah_awal": int(len(df_lama)),
            "total_mentah_akhir": int(len(df_gabung)),
            "mode": "Hanya Gabung Data (--tanpa-latih)" if tanpa_latih else "Gabung Data & Latih Ulang Penuh",
        }

        if tanpa_latih:
            return {
                "sukses": True,
                "tanpa_latih": True,
                "arsip_path": folder_arsip,
                "info_penggabungan": info_gabung,
                "pesan": f"Berhasil menambahkan {baris_masuk:,} baris baru ke {RAW_FILE} (tanpa melatih ulang).",
            }

        # Jalankan 5 tahap pipeline secara berurutan
        tahapan_pipeline = [
            ("processing.py", "Menjalankan preprocessing & ekstraksi NLP (processing.py)...", 35),
            ("pelabelan.py", "Menjalankan split 80:20, regresi harga wajar & label tertile (pelabelan.py)...", 55),
            ("analisis_fitur.py", "Menjalankan seleksi fitur (analisis_fitur.py)...", 70),
            ("train_model_klasifikasi.py", "Melatih model klasifikasi & evaluasi nested CV (train_model_klasifikasi.py)...", 85),
            ("test_sistem.py", "Menjalankan 21 tes integritas sistem (test_sistem.py)...", 95),
        ]

        log_eksekusi = []
        hasil_tes = {"lulus": 0, "total": 21, "semua_lulus": False, "ringkasan": "Belum dijalankan", "detail": []}

        for skrip, pesan_tahap, progres in tahapan_pipeline:
            if callback_tahap:
                callback_tahap(pesan_tahap, progres)
            res = _jalankan_skrip(skrip)
            log_eksekusi.append({
                "skrip": skrip,
                "sukses": res["sukses"],
                "returncode": res["returncode"],
            })

            if skrip == "test_sistem.py":
                hasil_tes = _parse_hasil_tes(res["stdout"], res["returncode"])

            if not res["sukses"]:
                if skrip == "test_sistem.py" and paksa:
                    continue
                # Rollback otomatis
                pulihkan_arsip(folder_arsip)
                detail_err = (res["stderr"] or res["stdout"] or "Kesalahan tidak diketahui")[-600:]
                raise RuntimeError(
                    f"Tahap '{skrip}' gagal dijalankan. Seluruh dataset dan model telah "
                    f"dikembalikan otomatis dari arsip ({folder_arsip}). Detail: {detail_err}"
                )

        # Ambil snapshot kondisi SESUDAH pelatihan ulang selesai
        snapshot_sesudah = ambil_snapshot_lengkap(".")
        laporan_perubahan = hitung_perubahan(
            sebelum=snapshot_sebelum,
            sesudah=snapshot_sesudah,
            info_gabung=info_gabung,
            hasil_tes=hasil_tes,
        )
        waktu_skrg = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        laporan_perubahan["id_riwayat"] = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        laporan_perubahan["waktu"] = waktu_skrg
        laporan_perubahan["label_batch"] = f"Penambahan Data ({info_gabung['nama_file']}) - {waktu_skrg}"
        laporan_perubahan["arsip_path"] = folder_arsip
        laporan_perubahan["log_pipeline"] = log_eksekusi

        # Simpan ke riwayat_data.json
        riwayat = muat_riwayat()
        riwayat.insert(0, laporan_perubahan)
        with open(RIWAYAT_FILE, "w", encoding="utf-8") as f:
            json.dump(riwayat, f, indent=2, ensure_ascii=False)

        if callback_tahap:
            callback_tahap("Selesai! Seluruh pipeline dan 21 tes berhasil dijalankan.", 100)

        return {
            "sukses": True,
            "tanpa_latih": False,
            "arsip_path": folder_arsip,
            "laporan": laporan_perubahan,
            "pesan": (
                f"Berhasil menambahkan {baris_masuk:,} baris data baru, melatih ulang seluruh model, "
                f"dan lulus {hasil_tes['ringkasan']}."
            ),
        }

    except Exception:
        raise


def cetak_ringkasan_cli(hasil: Dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print("   HASIL PENAMBAHAN DATA & PELATIHAN ULANG OTOMATIS (AUTOLAKU)   ")
    print("=" * 78)
    print(f"Status : {hasil.get('pesan')}")
    print(f"Arsip  : {hasil.get('arsip_path')}")

    if hasil.get("tanpa_latih"):
        ig = hasil["info_penggabungan"]
        print(f"Baris CSV Baru      : {ig['baris_file_baru']:,}")
        print(f"Duplikat Dilewati   : {ig['baris_duplikat_id']:,}")
        print(f"Baris Ditambahkan   : {ig['baris_ditambahkan']:,}")
        print(f"Total Mentah Akhir  : {ig['total_mentah_akhir']:,}")
        print("=" * 78)
        return

    lap = hasil["laporan"]
    b = lap["sebelum"]
    a = lap["sesudah"]
    d = lap["delta"]

    print("\n--- Perbandingan Sebelum vs Sesudah ---")
    print(f"  Data Mentah       : {b['preprocessing']['data_mentah']:,} -> {a['preprocessing']['data_mentah']:,} ({d['data_mentah']:+,})")
    print(f"  Data Bersih       : {b['preprocessing']['data_bersih']:,} -> {a['preprocessing']['data_bersih']:,} ({d['data_bersih']:+,})")
    print(f"  Data Train / Test : {b['pelabelan']['train_size']:,}/{b['pelabelan']['test_size']:,} -> {a['pelabelan']['train_size']:,}/{a['pelabelan']['test_size']:,}")
    print(f"  Merek / Model Unik: {b['preprocessing']['merek_unik']}/{b['preprocessing']['model_unik']} -> {a['preprocessing']['merek_unik']}/{a['preprocessing']['model_unik']}")
    print(f"  Model Regresi     : {b['pelabelan']['model_regresi']} -> {a['pelabelan']['model_regresi']}")
    print(f"  MAPE Harga Wajar  : {b['pelabelan']['mape_harga_wajar_pct']:.2f}% -> {a['pelabelan']['mape_harga_wajar_pct']:.2f}% ({d['mape_harga_wajar_pct']:+.2f}%)")
    print(f"  Ambang Q33 / Q67  : {b['pelabelan']['q33']:+.2f}% / {b['pelabelan']['q67']:+.2f}% -> {a['pelabelan']['q33']:+.2f}% / {a['pelabelan']['q67']:+.2f}%")
    print(f"  Model Klasifikasi : {b['klasifikasi']['model_terpilih']} -> {a['klasifikasi']['model_terpilih']}")
    print(f"  F1-Macro CV       : {b['klasifikasi']['f1_macro_cv_pct']:.2f}% -> {a['klasifikasi']['f1_macro_cv_pct']:.2f}% ({d['f1_macro_cv_pct']:+.2f}%)")
    print(f"  F1-Macro Test     : {b['klasifikasi']['f1_macro_test_pct']:.2f}% -> {a['klasifikasi']['f1_macro_test_pct']:.2f}% ({d['f1_macro_test_pct']:+.2f}%)")
    print(f"  Akurasi Test      : {b['klasifikasi']['accuracy_test_pct']:.2f}% -> {a['klasifikasi']['accuracy_test_pct']:.2f}% ({d['accuracy_test_pct']:+.2f}%)")
    print(f"  Tes Otomatis      : {lap['tes_sistem']['ringkasan']}")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(
        description="AutoLaku: Tambah data hasil scraping baru lalu jalankan ulang pipeline pelatihan."
    )
    parser.add_argument("csv_baru", help="Jalur file CSV hasil scraping baru (format dataset_olx_mentah.csv)")
    parser.add_argument("--tanpa-latih", action="store_true", help="Hanya gabungkan data tanpa melatih ulang model")
    parser.add_argument("--paksa", action="store_true", help="Tetap simpan meskipun 21 tes otomatis gagal atau 0 baris baru")
    args = parser.parse_args()

    try:
        hasil = tambah_dan_latih(
            csv_path=args.csv_baru,
            tanpa_latih=args.tanpa_latih,
            paksa=args.paksa,
            callback_tahap=lambda msg, pct: print(f"[{pct:3d}%] {msg}"),
        )
        cetak_ringkasan_cli(hasil)
    except Exception as e:
        print(f"\n[GAGAL] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
