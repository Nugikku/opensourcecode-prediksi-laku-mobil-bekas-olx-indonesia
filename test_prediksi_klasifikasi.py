"""
test_prediksi_klasifikasi.py
--------------------------------------------------------------------------------
AutoLaku: Uji Coba & Antarmuka Terminal (CLI) Interaktif untuk Prediksi Kecepatan Laku
--------------------------------------------------------------------------------
Berinteraksi langsung dengan mesin_prediksi.py untuk menguji skenario input penjual.
"""

import sys
from mesin_prediksi import muat_sistem, daftar_pilihan, prediksi, InputTidakValid
from processing import AMBANG_KEYAKINAN_SAMPEL

def format_rupiah(nominal):
    return f"Rp {int(nominal):,}".replace(",", ".")

def main():
    print("=" * 80)
    print("   AUTOLAKU: SISTEM PREDIKSI KECEPATAN LAKU MOBIL BEKAS OLX (CLI MODE)   ")
    print("=" * 80)

    print("\n[INFO] Memuat model machine learning dan dataset...")
    try:
        sistem = muat_sistem()
    except FileNotFoundError as e:
        print(f"\n[ERROR] Gagal memuat sistem: {e}")
        sys.exit(1)

    df_pilihan = daftar_pilihan(sistem)
    merek_list = sorted(df_pilihan['merek'].unique())

    print(f"[SUKSES] Sistem siap! Total merek tersedia: {len(merek_list)} merek.\n")

    print("-" * 80)
    print("📝 SIMULASI INPUT DATA PENJUAL")
    print("-" * 80)

    # 1. Pilih Merek
    print("Daftar Merek Populer:", ", ".join(merek_list[:10]), "...")
    merek_input = input("Masukkan Merek Mobil (contoh: Toyota): ").strip()
    
    # Validasi Merek
    model_tersedia = df_pilihan[df_pilihan['merek'].str.lower() == merek_input.lower()]
    if model_tersedia.empty:
        print(f"⚠️ Merek '{merek_input}' tidak ditemukan dalam dataset. Menggunakan default pertama.")
        merek_input = merek_list[0]
        model_tersedia = df_pilihan[df_pilihan['merek'] == merek_input]

    # 2. Pilih Model
    model_list = sorted(model_tersedia['model'].unique())
    print(f"\nModel tersedia untuk {merek_input}:", ", ".join(model_list[:10]), "...")
    model_input = input(f"Masukkan Model/Seri ({merek_input}): ").strip()
    if not model_input and len(model_list) > 0:
        model_input = model_list[0]

    # 3. Input Tahun, KM, Harga, Transmisi, dll.
    try:
        tahun_input = int(input("Tahun Perakitan (misal: 2018): Critical [1995-2026] -> ") or 2018)
        km_input = int(input("Jarak Tempuh / Odometer (KM, misal: 65000): ") or 65000)
        harga_input = float(input("Rencana Harga Jual (Rp, misal: 150000000): ") or 150_000_000)
    except ValueError:
        print("⚠️ Input angka tidak valid. Menggunakan nilai default.")
        tahun_input, km_input, harga_input = 2018, 65_000, 150_000_000

    transmisi_input = input("Transmisi ([A]utomatic / [M]anual): ").strip().lower()
    transmisi = "Manual" if transmisi_input.startswith("m") else "Automatic"

    tipe_penjual_input = input("Profil Penjual ([I]ndividu / [D]ealer): ").strip().lower()
    tipe_penjual = "Dealer" if tipe_penjual_input.startswith("d") else "Individu"

    deskripsi_input = input("\nRencana Deskripsi Iklan (teks bebas):\n> ").strip()
    if not deskripsi_input:
        deskripsi_input = "Mobil terawat, pajak hidup, service record resmi, bebas banjir dan tidak pernah tabrakan."

    print("\n" + "=" * 80)
    print("⏳ Menjalankan Mesin Prediksi & Analisis NLP...")
    print("=" * 80)

    try:
        hasil = prediksi(
            sistem=sistem,
            merek=merek_input,
            model=model_input,
            tahun=tahun_input,
            transmisi=transmisi,
            km=km_input,
            harga_iklan=harga_input,
            deskripsi=deskripsi_input,
            tipe_penjual=tipe_penjual
        )
    except InputTidakValid as e:
        print(f"\n❌ Validasi Gagal: {e}")
        sys.exit(1)

    # Tampilkan Hasil
    print(f"\n📊 HASIL PREDIKSI KENDARAAN: {hasil['merek']} {hasil['model']} ({hasil['tahun']})")
    print("-" * 80)

    for p in hasil['peringatan']:
        print(f"⚠️ PERINGATAN: {p}")

    print(f"• Rencana Harga Jual     : {format_rupiah(hasil['harga_iklan'])}")
    print(f"• Estimasi Harga Wajar   : {format_rupiah(hasil['harga_wajar'])} ({hasil['sumber_harga_wajar']})")
    print(f"• Deviasi Harga          : {hasil['deviasi_persen']:+.2f}%")
    print(f"• Total Odometer         : {hasil['km']:,} KM ({hasil['km_per_tahun']:,.0f} KM/tahun)")
    print(f"• Garansi Terdeteksi     : {hasil['ada_garansi']}")
    print(f"• Indikasi Kredit / TDP  : {'Ya' if hasil['indikasi_kredit_tdp'] else 'Tidak'}")
    print(f"• Data Pembanding Setara : {hasil['jumlah_sampel']} unit (Ambang: {AMBANG_KEYAKINAN_SAMPEL})")

    print("\n" + "-" * 40)
    status = hasil['status']
    if status == "Cepat":
        print("🎯 STATUS PREDIKSI : ✅ CEPAT LAKU (< 15 Hari)")
    elif status == "Sedang":
        print("🎯 STATUS PREDIKSI : 🟡 SEDANG LAKU (15 - 30 Hari)")
    else:
        print("🎯 STATUS PREDIKSI : 🔴 LAMBAT LAKU (> 30 Hari)")
    print("-" * 40)

    if hasil['proba']:
        print("\n📈 Probabilitas Tiap Kelas Likuiditas:")
        for kelas, prob in hasil['proba'].items():
            print(f"   - {kelas.ljust(8)}: {prob * 100:.2f}%")

    if hasil['keyakinan_rendah']:
        print("\n⚠️ Catatan: Keyakinan model berada pada tingkat moderat karena ukuran sampel historis terbatas.")

    print("\n" + "=" * 80)
    print("Selesai. Gunakan app.py untuk menjalankan antarmuka web interaktif.")
    print("=" * 80)

if __name__ == "__main__":
    main()