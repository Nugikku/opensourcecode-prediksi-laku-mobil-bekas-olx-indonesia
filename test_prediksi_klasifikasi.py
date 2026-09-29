"""
AutoLaku: Sistem Prediksi Kecepatan Laku Mobil Bekas OLX (antarmuka terminal untuk penjual).
Logika prediksi ada di mesin_prediksi.py (dipakai bersama app.py dan test_sistem.py).
"""
from mesin_prediksi import (muat_sistem, daftar_pilihan, prediksi, InputTidakValid,
                            validasi_tahun, validasi_transmisi, validasi_km, validasi_harga)
from processing import TAHUN_SEKARANG

def tanya(prompt, validator):
    while True:
        try:
            return validator(input(prompt))
        except InputTidakValid as e:
            print(f"[!] {e}")

def pilih_nomor(judul, daftar):
    print(f"\n{judul}")
    for i in range(0, len(daftar), 4):
        print("".join(f"[{i + j + 1:<2}] {daftar[i + j]:<16} " for j in range(4) if i + j < len(daftar)))
    while True:
        try:
            no = int(input(f"\nPilih nomor (1-{len(daftar)}): ").strip())
            if 1 <= no <= len(daftar):
                return daftar[no - 1]
        except ValueError:
            pass
        print("[!] Masukkan nomor yang tersedia.")

try:
    sistem = muat_sistem()
except FileNotFoundError as e:
    print(f"[!] {e}")
    raise SystemExit(1)

print("=" * 80)
print("   AUTOLAKU: SISTEM PREDIKSI KECEPATAN LAKU MOBIL BEKAS OLX (PENJUAL)   ")
print("=" * 80)

pilihan = daftar_pilihan(sistem)
merek = pilih_nomor("PILIH MEREK KENDARAAN:", sorted(pilihan['merek'].unique()))
model = pilih_nomor(f"PILIH MODEL {merek.upper()}:", sorted(pilihan[pilihan['merek'] == merek]['model'].unique()))

print("\n" + "=" * 80 + "\nFORMULIR INPUT PENJUAL\n" + "=" * 80)
tahun = tanya(f"Tahun mobil (1995-{TAHUN_SEKARANG}): ", validasi_tahun)
transmisi = tanya("Transmisi (manual/automatic): ", validasi_transmisi)
km = tanya("Kilometer (mis. 85000): ", validasi_km)
harga = tanya("Rencana harga jual (mis. 150jt atau 150000000): ", validasi_harga)
deskripsi = input("Rencana deskripsi iklan:\n> ").strip()

h = prediksi(sistem, merek, model, tahun, transmisi, km, harga, deskripsi)

print("\n" + "=" * 80 + "\n                    AUTOLAKU: HASIL PREDIKSI                    \n" + "=" * 80)
print(f"Unit               : {merek} {model} ({h['tahun']}) | {h['transmisi'].title()}")
print(f"Odometer           : {h['km']:,} KM ({h['km_per_tahun']:,} KM/Tahun)")
print(f"Garansi/Riwayat    : {h['ada_garansi']} | Indikasi TDP/Kredit: {'Ya' if h['indikasi_kredit_tdp'] else 'Tidak'}")
print("-" * 80)
print(f"Rencana Harga Jual : Rp {int(h['harga_iklan']):,}")
print(f"Harga Wajar        : Rp {int(h['harga_wajar']):,}  ({h['sumber_harga_wajar']})")
print(f"Deviasi Harga      : {h['deviasi_persen']:+.1f}%")
print("-" * 80)
print(f"PREDIKSI           : [{h['status'].upper()} LAKU]")
if h['proba']:
    print("Peluang kelas      : " + " | ".join(f"{k}: {v * 100:.1f}%" for k, v in h['proba'].items()))
print(f"Data pembanding    : {h['jumlah_sampel']} unit sejenis")
for p in h['peringatan']:
    print(f"[!] {p}")
print("=" * 80)