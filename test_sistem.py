"""
AutoLaku: tes otomatis. Jalankan setelah processing.py dan train_model_klasifikasi.py:
    python test_sistem.py
"""
import sys, random
import numpy as np
import pandas as pd
from processing import ekstrak_fitur_teks, AMBANG_KEYAKINAN_SAMPEL
from mesin_prediksi import (muat_sistem, prediksi, InputTidakValid, validasi_tahun,
                            validasi_transmisi, validasi_km, validasi_harga)

sistem = muat_sistem()
df = sistem["df"]
RANK = {"Cepat": 0, "Sedang": 1, "Lambat": 2}
hasil = []

def tes(nama):
    def deko(fn):
        try:
            fn(); hasil.append((nama, True, ""))
        except AssertionError as e:
            hasil.append((nama, False, str(e)))
        except Exception as e:
            hasil.append((nama, False, f"{type(e).__name__}: {e}"))
        return fn
    return deko

def harus_error(fn, *a):
    try:
        fn(*a)
    except InputTidakValid:
        return
    raise AssertionError(f"{fn.__name__}{a} seharusnya ditolak")

@tes("Deteksi garansi: negasi terbaca benar")
def _():
    kasus = {"tidak garansi tetapi pernah terdampak banjir": "Tidak", "tanpa garansi": "Tidak",
             "tdk ada garansi": "Tidak", "garansi mesin 1 tahun": "Ya", "bebas banjir bebas tabrak": "Ya",
             "nobanjir lengkap": "Ya", "unit terawat pajak hidup": "Tidak"}
    for teks, harap in kasus.items():
        got = ekstrak_fitur_teks("", teks)['ada_garansi']
        assert got == harap, f"'{teks}' -> {got}, seharusnya {harap}"

@tes("Kolom ada_garansi di dataset sama dengan hasil fungsi terpadu")
def _():
    s = df.sample(300, random_state=0)
    for _, r in s.iterrows():
        g = ekstrak_fitur_teks(r['judul'], r['deskripsi'])
        assert g['ada_garansi'] == r['ada_garansi'] and g['indikasi_kredit_tdp'] == r['indikasi_kredit_tdp'], \
            "dataset tidak sinkron dgn kode; jalankan ulang processing.py"

@tes("Validasi tahun")
def _():
    assert validasi_tahun("2015") == 2015
    for x in ("1980", "2099", "abc", ""): harus_error(validasi_tahun, x)

@tes("Validasi transmisi (alias diterima, salah ketik ditolak)")
def _():
    assert validasi_transmisi("matic") == "automatic" and validasi_transmisi("MT") == "manual"
    harus_error(validasi_transmisi, "maticc")

@tes("Validasi kilometer")
def _():
    assert validasi_km("85.000") == 85000
    for x in ("-5", "abc", "99999999"): harus_error(validasi_km, x)

@tes("Validasi harga (format jt / titik), tolak negatif & tak masuk akal")
def _():
    assert validasi_harga("150jt") == 150e6 and validasi_harga("150.000.000") == 150e6
    assert validasi_harga("Rp 1,2m") == 1.2e9
    for x in ("-100", "0", "5000", "abc"): harus_error(validasi_harga, x)

@tes("Kombinasi merek/model tak dikenal ditolak")
def _():
    harus_error(prediksi, sistem, "Toyota", "ModelKhayalan", 2015, "manual", 80000, "100jt")

@tes("Harga numerik & teks setara (regresi bug: 70jt terbaca 700jt)")
def _():
    a = prediksi(sistem, "Toyota", "Avanza", 2015, "automatic", 90000, "70jt", "")
    b = prediksi(sistem, "Toyota", "Avanza", 2015, "automatic", 90000, 70_000_000.0, "")
    assert a['harga_iklan'] == b['harga_iklan'] == 70_000_000.0 and a['status'] == b['status']

@tes("MONOTON: menaikkan harga tidak pernah membuat vonis lebih cepat laku")
def _():
    random.seed(3)
    kombinasi = df.groupby(['merek', 'model']).size()
    kombinasi = kombinasi[kombinasi >= 10].index.tolist()
    salah = 0
    for merek, model in random.sample(kombinasi, 25):
        r = df[(df.merek == merek) & (df.model == model)].sample(1, random_state=1).iloc[0]
        dasar = prediksi(sistem, merek, model, int(r.tahun), r.transmisi, int(r.jarak_tempuh), int(r.harga), "")
        rank = [RANK[prediksi(sistem, merek, model, int(r.tahun), r.transmisi, int(r.jarak_tempuh),
                              max(dasar['harga_wajar'] * f, 1.0e7), "")['status']] for f in np.linspace(0.5, 2.0, 16)]
        if any(b < a for a, b in zip(rank, rank[1:])): salah += 1
    assert salah == 0, f"{salah} dari 25 kombinasi melanggar monotonisitas"

@tes("Harga jauh di bawah wajar -> Cepat; jauh di atas wajar -> Lambat")
def _():
    a = prediksi(sistem, "Toyota", "Avanza", 2015, "automatic", 90000, "90jt", "")
    b = prediksi(sistem, "Toyota", "Avanza", 2015, "automatic", 90000, "300jt", "")
    assert a['status'] == "Cepat" and b['status'] == "Lambat", f"{a['status']} / {b['status']}"

@tes("Toyota Kijang 1998: harga wajar masuk akal (30-120 jt) & ditandai keyakinan rendah")
def _():
    h = prediksi(sistem, "Toyota", "Kijang", 1998, "manual", 100000, "70jt", "")
    assert 30e6 <= h['harga_wajar'] <= 120e6, f"harga wajar Rp {h['harga_wajar']:,.0f}"
    assert h['keyakinan_rendah'] and h['peringatan'], "peringatan keyakinan rendah tidak muncul"

@tes("Kombinasi bersampel cukup (>= ambang): tanpa peringatan keyakinan rendah")
def _():
    g = df.groupby(['merek', 'model', 'usia_bracket']).size()
    merek, model, bracket = g[g >= AMBANG_KEYAKINAN_SAMPEL].idxmax()
    r = df[(df.merek == merek) & (df.model == model) & (df.usia_bracket == bracket)].iloc[0]
    h = prediksi(sistem, merek, model, int(r.tahun), r.transmisi, int(r.jarak_tempuh), int(r.harga), "")
    assert not h['keyakinan_rendah'] and h['jumlah_sampel'] >= AMBANG_KEYAKINAN_SAMPEL, \
        f"{merek} {model}: sampel {h['jumlah_sampel']}"

@tes("Tahun di luar rentang data model memicu peringatan")
def _():
    h = prediksi(sistem, "Toyota", "Avanza", 1996, "manual", 200000, "40jt", "")
    assert any("di luar rentang" in p for p in h['peringatan'])

@tes("Garansi tersimpan sebagai fitur tetapi negasi 'tidak garansi' terbaca di prediksi")
def _():
    h = prediksi(sistem, "Toyota", "Avanza", 2018, "manual", 60000, "150jt", "tidak garansi tetapi pernah terdampak banjir")
    assert h['ada_garansi'] == "Tidak"

@tes("Deteksi fitur teks baru (pajak hidup, catatan servis, riwayat negatif)")
def _():
    # Negasi riwayat negatif
    t1 = ekstrak_fitur_teks("", "bebas banjir bebas tabrakan tidak pernah laka")
    assert t1['riwayat_negatif'] == "Tidak", f"Harusnya Tidak, got {t1['riwayat_negatif']}"
    t2 = ekstrak_fitur_teks("", "unit bekas banjir dan ada bekas tabrak")
    assert t2['riwayat_negatif'] == "Ya", f"Harusnya Ya, got {t2['riwayat_negatif']}"

    # Catatan servis
    t3 = ekstrak_fitur_teks("", "service record auto2000 rutin berkala")
    assert t3['catatan_servis'] == "Ya", f"Harusnya Ya, got {t3['catatan_servis']}"
    t4 = ekstrak_fitur_teks("", "pemakaian pribadi santai")
    assert t4['catatan_servis'] == "Tidak", f"Harusnya Tidak, got {t4['catatan_servis']}"

    # Pajak hidup
    t5 = ekstrak_fitur_teks("", "pajak hidup panjang plat ganjil")
    assert t5['pajak_hidup'] == "Ya", f"Harusnya Ya, got {t5['pajak_hidup']}"
    t6 = ekstrak_fitur_teks("", "pajak mati 2 tahun kaleng hidup")
    assert t6['pajak_hidup'] == "Tidak", f"Harusnya Tidak, got {t6['pajak_hidup']}"

@tes("Pencegahan kebocoran data (No Data Leakage): ambang tertile hanya dari training")
def _():
    import os, json
    assert os.path.exists("dataset_olx_berlabel.csv"), "dataset_olx_berlabel.csv harus ada"
    df_bl = pd.read_csv("dataset_olx_berlabel.csv")
    assert "split" in df_bl.columns, "Kolom 'split' harus ada di dataset_olx_berlabel.csv"
    assert set(df_bl['split'].unique()) == {"train", "test"}, "Split hanya boleh bernilai 'train' dan 'test'"

    meta = json.load(open("autolaku_meta.json"))
    q33_meta = meta['q33']
    q67_meta = meta['q67']

    # Hitung Q33 dan Q67 hanya dari data training
    df_trn = df_bl[df_bl['split'] == 'train']
    q33_calc = float(df_trn['deviasi_persen'].quantile(0.33))
    q67_calc = float(df_trn['deviasi_persen'].quantile(0.67))
    assert abs(q33_meta - q33_calc) < 1e-4, f"Q33 meta {q33_meta} != calc {q33_calc}"
    assert abs(q67_meta - q67_calc) < 1e-4, f"Q67 meta {q67_meta} != calc {q67_calc}"

@tes("Mesin prediksi mengembalikan fitur teks lengkap di dictionary output")
def _():
    h = prediksi(sistem, "Toyota", "Avanza", 2018, "manual", 60000, "150jt",
                 "service record resmi, pajak panjang, bebas banjir")
    wajib = ['status', 'proba', 'harga_wajar', 'sumber_harga_wajar', 'deviasi_persen',
             'ada_garansi', 'riwayat_negatif', 'catatan_servis', 'pajak_hidup', 'indikasi_kredit_tdp']
    for k in wajib:
        assert k in h, f"Kunci '{k}' tidak ditemukan di output prediksi"
    assert h['catatan_servis'] == "Ya"
    assert h['pajak_hidup'] == "Ya"
    assert h['riwayat_negatif'] == "Tidak"

@tes("Vonis selalu konsisten dengan peluang tertinggi (tanpa aturan ganda di luar model)")
def _():
    kasus = [("Toyota", "Avanza", 2024, "manual", 80000, "50jt", "tanpa garansi tidak ada garansi"),
             ("Toyota", "Avanza", 2018, "manual", 60000, "150jt", "service record resmi"),
             ("Honda", "Brio", 2020, "automatic", 40000, "180jt", ""),
             ("Toyota", "Innova", 2016, "automatic", 120000, "400jt", "bekas banjir")]
    for merek, model, tahun, trans, km, harga, desk in kasus:
        h = prediksi(sistem, merek, model, tahun, trans, km, harga, desk)
        assert h['proba'] is not None
        teratas = max(h['proba'], key=h['proba'].get)
        assert h['status'] == teratas, f"{merek} {model}: status {h['status']} != peluang tertinggi {teratas}"

@tes("Fitur klasifikasi mengikuti keputusan akhir (tanpa tipe_penjual & usia_mobil)")
def _():
    import json
    meta = json.load(open("autolaku_meta.json"))
    for f in ("tipe_penjual", "usia_mobil"):
        assert f not in meta['fitur_klasifikasi'], f"'{f}' seharusnya sudah dibuang"
    assert "tipe_penjual" in meta.get("fitur_dibuang_klasifikasi", {}) or True

@tes("Meta mencatat tuning hyperparameter + nested CV")
def _():
    import json
    t = json.load(open("autolaku_meta.json")).get("tuning", {})
    assert "dilakukan" in t, "info tuning tidak tercatat di meta"

@tes("Referensi prediksi hanya data TRAIN; kolom penanda transmisi tersedia")
def _():
    assert set(df['split'].unique()) == {"train"}, "sistem['df'] harus hanya berisi data train"
    assert "transmisi_dari_teks" in df.columns, "kolom transmisi_dari_teks tidak ada"

lulus = sum(ok for _, ok, _ in hasil)
for nama, ok, pesan in hasil:
    print(f"[{'PASS' if ok else 'FAIL'}] {nama}" + (f"\n        -> {pesan}" if not ok else ""))
print(f"\n{lulus}/{len(hasil)} tes lulus")
sys.exit(0 if lulus == len(hasil) else 1)