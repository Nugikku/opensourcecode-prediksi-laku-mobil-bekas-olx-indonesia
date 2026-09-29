"""
AutoLaku: mesin prediksi (SATU sumber logika inferensi).
Dipakai oleh test_prediksi_klasifikasi.py (CLI), test_sistem.py, dan app.py:

    from mesin_prediksi import muat_sistem, prediksi
    sistem = muat_sistem()
    hasil = prediksi(sistem, "Toyota", "Avanza", 2015, "matic", 90000, "130jt", "terawat, pajak hidup")
"""

import os, re, json, pickle
import numpy as np
import pandas as pd
from processing import (TAHUN_SEKARANG, AMBANG_KEYAKINAN_SAMPEL, ekstrak_fitur_teks,
                        pilih_harga_wajar, klip_deviasi)

TAHUN_MIN = 1995
HARGA_MIN, HARGA_MAX = 10_000_000, 10_000_000_000
KM_MAX = 1_500_000

class InputTidakValid(ValueError):
    """Dilempar bila masukan pengguna tidak masuk akal."""

# ---------------- validasi & normalisasi masukan ----------------
def validasi_tahun(nilai):
    try:
        tahun = int(str(nilai).strip())
    except ValueError:
        raise InputTidakValid("Tahun harus berupa angka, mis. 2015.")
    if not TAHUN_MIN <= tahun <= TAHUN_SEKARANG:
        raise InputTidakValid(f"Tahun harus antara {TAHUN_MIN} dan {TAHUN_SEKARANG}.")
    return tahun

def validasi_transmisi(nilai):
    t = str(nilai).strip().lower()
    if t in {"automatic", "otomatis", "matic", "at", "a/t", "cvt", "auto", "automatik"}:
        return "automatic"
    if t in {"manual", "mt", "m/t"}:
        return "manual"
    raise InputTidakValid("Transmisi harus 'manual' atau 'automatic' (boleh: matic, at, mt).")

def validasi_km(nilai):
    t = re.sub(r'[.,\s]|km$', '', str(nilai).strip().lower())
    if not t.isdigit():
        raise InputTidakValid("Kilometer harus berupa angka, mis. 85000.")
    km = int(t)
    if km > KM_MAX:
        raise InputTidakValid(f"Kilometer maksimal {KM_MAX:,}.")
    return km

def validasi_harga(nilai):
    """Terima '150000000', '150.000.000', '150jt', '150,5jt', '1.2m' (miliar)."""
    if isinstance(nilai, (int, float, np.integer, np.floating)):
        harga = float(nilai)          # sudah berupa angka: jangan diparse sebagai teks
        if not HARGA_MIN <= harga <= HARGA_MAX:
            raise InputTidakValid(f"Harga harus antara Rp {HARGA_MIN:,} dan Rp {HARGA_MAX:,}.")
        return harga
    t = str(nilai).strip().lower().replace('rp', '').replace(' ', '')
    m = re.fullmatch(r'(\d[\d.,]*)(jt|juta|m|miliar)?', t)
    if not m:
        raise InputTidakValid("Harga tidak dikenali. Contoh: 150000000, 150.000.000, atau 150jt.")
    angka, satuan = m.groups()
    if satuan:
        harga = float(angka.replace(',', '.')) * (1e6 if satuan in ('jt', 'juta') else 1e9)
    else:
        harga = float(re.sub(r'[.,]', '', angka))
    if not HARGA_MIN <= harga <= HARGA_MAX:
        raise InputTidakValid(f"Harga harus antara Rp {HARGA_MIN:,} dan Rp {HARGA_MAX:,}.")
    return harga

# ---------------- muat sistem ----------------
def muat_sistem(folder="."):
    p = lambda n: os.path.join(folder, n)
    wajib = ["model_klasifikasi_likuiditas.pkl", "model_regresi_harga.pkl", "autolaku_meta.json", "dataset_olx_bersih.csv"]
    hilang = [n for n in wajib if not os.path.exists(p(n))]
    if hilang:
        raise FileNotFoundError(f"File belum ada: {hilang}. Jalankan processing.py lalu train_model_klasifikasi.py.")
    with open(p("model_klasifikasi_likuiditas.pkl"), "rb") as f: klasifikasi = pickle.load(f)
    with open(p("model_regresi_harga.pkl"), "rb") as f: regresi = pickle.load(f)
    meta = json.load(open(p("autolaku_meta.json")))
    df = pd.read_csv(p("dataset_olx_bersih.csv"))
    return {"klasifikasi": klasifikasi, "regresi": regresi, "meta": meta, "df": df}

def daftar_pilihan(sistem):
    """DataFrame merek+model yang boleh dipilih penjual."""
    df = sistem["df"]
    ok = df[(df['model'].fillna('').str.lower() != 'lainnya') & (df['merek'].fillna('').str.lower() != 'lainnya')
            & (df['model'].fillna('').str.strip() != '')]
    return ok

# ---------------- prediksi ----------------
def prediksi(sistem, merek, model, tahun, transmisi, km, harga_iklan, deskripsi="", tipe_penjual="Individu"):
    tahun = validasi_tahun(tahun)
    transmisi = validasi_transmisi(transmisi)
    km = validasi_km(km)
    harga_iklan = validasi_harga(harga_iklan)

    df = sistem["df"]
    sub = df[(df['merek'] == merek) & (df['model'] == model)]
    if sub.empty:
        raise InputTidakValid(f"Kombinasi '{merek} {model}' tidak ada di data.")

    usia = max(1, TAHUN_SEKARANG - tahun)
    km_per_tahun = round(km / usia)
    teks = ekstrak_fitur_teks("", deskripsi)

    dasar = pd.DataFrame([{
        'merek': merek, 'model': model, 'transmisi': transmisi, 'tipe_penjual': tipe_penjual,
        'ada_garansi': teks['ada_garansi'], 'tahun': tahun, 'jarak_tempuh': km,
        'usia_mobil': usia, 'km_per_tahun': km_per_tahun}])

    harga_model = float(sistem["regresi"].predict(dasar)[0])
    bracket = (usia // 5) * 5
    sampel = int((sub['usia_bracket'] == bracket).sum())
    harga_wajar, sumber = pilih_harga_wajar(harga_model, sampel, df, merek, model, usia)

    deviasi = klip_deviasi((harga_iklan - harga_wajar) / harga_wajar * 100)
    status = sistem["klasifikasi"].predict(dasar.assign(deviasi_persen=deviasi))[0]
    proba = None
    if hasattr(sistem["klasifikasi"].named_steps['clf'], "predict_proba"):
        pr = sistem["klasifikasi"].predict_proba(dasar.assign(deviasi_persen=deviasi))[0]
        proba = dict(zip(sistem["klasifikasi"].named_steps['clf'].classes_, map(float, pr)))

    peringatan = []
    if sampel < AMBANG_KEYAKINAN_SAMPEL:
        peringatan.append(f"Hanya {sampel} data pembanding untuk model & rentang usia ini (ambang {AMBANG_KEYAKINAN_SAMPEL}); "
                          f"harga wajar dihitung dengan {sumber}. Anggap sebagai referensi kasar.")
    if tahun < sub['tahun'].min() or tahun > sub['tahun'].max():
        peringatan.append(f"Tahun {tahun} di luar rentang data model ini ({int(sub['tahun'].min())}-{int(sub['tahun'].max())}).")
    if km_per_tahun > 60_000:
        peringatan.append(f"Pemakaian {km_per_tahun:,} km/tahun sangat tinggi; periksa kembali angka kilometer.")
    if km < 1000 and usia > 3:
        peringatan.append("Kilometer sangat rendah untuk usia mobil ini; periksa kembali angka kilometer.")
    if harga_iklan < 0.3 * harga_wajar or harga_iklan > 3 * harga_wajar:
        peringatan.append("Harga jual sangat jauh dari harga wajar; periksa kembali angka yang diketik.")

    return {"status": status, "proba": proba, "harga_wajar": harga_wajar, "sumber_harga_wajar": sumber,
            "deviasi_persen": deviasi, "harga_iklan": harga_iklan, "km_per_tahun": km_per_tahun,
            "ada_garansi": teks['ada_garansi'], "indikasi_kredit_tdp": teks['indikasi_kredit_tdp'],
            "jumlah_sampel": sampel, "keyakinan_rendah": sampel < AMBANG_KEYAKINAN_SAMPEL,
            "peringatan": peringatan, "usia_mobil": usia, "transmisi": transmisi, "tahun": tahun, "km": km}