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

@tes("Toyota Avanza 2015 (data cukup): tanpa peringatan sampel")
def _():
    h = prediksi(sistem, "Toyota", "Avanza", 2015, "automatic", 90000, "130jt", "")
    assert not h['keyakinan_rendah'] and h['jumlah_sampel'] >= AMBANG_KEYAKINAN_SAMPEL

@tes("Tahun di luar rentang data model memicu peringatan")
def _():
    h = prediksi(sistem, "Toyota", "Avanza", 1996, "manual", 200000, "40jt", "")
    assert any("di luar rentang" in p for p in h['peringatan'])

@tes("Garansi tersimpan sebagai fitur tetapi negasi 'tidak garansi' terbaca di prediksi")
def _():
    h = prediksi(sistem, "Toyota", "Avanza", 2018, "manual", 60000, "150jt", "tidak garansi tetapi pernah terdampak banjir")
    assert h['ada_garansi'] == "Tidak"

lulus = sum(ok for _, ok, _ in hasil)
for nama, ok, pesan in hasil:
    print(f"[{'PASS' if ok else 'FAIL'}] {nama}" + (f"\n        -> {pesan}" if not ok else ""))
print(f"\n{lulus}/{len(hasil)} tes lulus")
sys.exit(0 if lulus == len(hasil) else 1)