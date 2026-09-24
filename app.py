import os
import re
import pickle
import pandas as pd
import numpy as np
import streamlit as st

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="AutoLiquid DSS - Smart Car Pricing & Liquidity",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

FILE_MODEL = "model_regresi_harga.pkl"
FILE_DATA = "dataset_olx_bersih.csv"

# ============================================================
# MEMUAT MODEL & DATASET
# ============================================================
@st.cache_resource
def muat_model():
    if not os.path.exists(FILE_MODEL):
        return None
    with open(FILE_MODEL, "rb") as f:
        return pickle.load(f)

@st.cache_data
def muat_dataset():
    if not os.path.exists(FILE_DATA):
        return None
    df = pd.read_csv(FILE_DATA)
    df_valid = df[
        (df['model'].fillna('').str.lower() != 'lainnya') &
        (df['merek'].fillna('').str.lower() != 'lainnya') &
        (df['model'].fillna('').str.strip() != '') &
        (df['harga'].notna()) &
        (df['tahun'].notna()) &
        (df['transmisi'].notna())
    ].copy().reset_index(drop=True)
    return df_valid

model_pipeline = muat_model()
df = muat_dataset()

if model_pipeline is None or df is None:
    st.error("File 'model_regresi_harga.pkl' atau 'dataset_olx_bersih.csv' belum ditemukan. Jalankan processing.py dan train_model_regresi.py terlebih dahulu.")
    st.stop()

# ============================================================
# FUNGSI VALUASI & EVALUASI LIKUIDITAS
# ============================================================
def evaluasi_likuiditas(merek, model, tahun, transmisi, km, tipe_penjual, ada_garansi, harga_input, indikasi_tdp):
    usia_mobil = max(1, 2026 - tahun)
    km_per_tahun = round(km / usia_mobil)

    data_tes = pd.DataFrame([{
        'merek': merek,
        'model': model,
        'transmisi': transmisi.lower(),
        'tipe_penjual': tipe_penjual,
        'ada_garansi': ada_garansi,
        'tahun': tahun,
        'jarak_tempuh': km,
        'usia_mobil': usia_mobil,
        'km_per_tahun': km_per_tahun
    }])
    
    pred_mentah = float(model_pipeline.predict(data_tes)[0])
    BATAS_MINIMUM = 20_000_000.0

    # Pengaman ekstrapolasi linier untuk mobil tua
    if pred_mentah < BATAS_MINIMUM:
        df_sejenis = df[(df['merek'] == merek) & (df['model'] == model)]
        if not df_sejenis.empty:
            harga_wajar = max(BATAS_MINIMUM, float(df_sejenis['harga'].median() * 0.55))
        else:
            harga_wajar = BATAS_MINIMUM
    else:
        harga_wajar = pred_mentah

    deviasi = ((harga_input - harga_wajar) / harga_wajar) * 100
    is_jebakan = (deviasi <= -35.0 and indikasi_tdp == 1)

    if deviasi <= -3.0:
        if km_per_tahun > 30000 and ada_garansi == 'Tidak':
            status = "Sedang"
            alasan = "Harga di bawah pasar, namun intensitas kilometer pemakaian tinggi tanpa garansi."
        else:
            status = "Cepat"
            alasan = "Harga penawaran sangat atraktif di bawah estimasi wajar pasar."
    elif -3.0 < deviasi <= 7.0:
        status = "Sedang"
        alasan = "Harga penawaran seimbang dengan nilai rata-rata pasar."
    else:
        status = "Lambat"
        alasan = "Harga penawaran melebihi harga wajar pasar (overpriced)."

    return harga_wajar, deviasi, status, alasan, is_jebakan, usia_mobil, km_per_tahun

# ============================================================
# SIDEBAR: SELEKSI PERSONA
# ============================================================
st.sidebar.title("🚗 AutoLiquid DSS")
st.sidebar.caption("Hybrid Decision Support System: ML Regression + Liquidity Engine")

persona = st.sidebar.radio(
    "Sudut Pandang Pengguna:",
    ("Penjual (Input Iklan Mandiri & Uji Likuiditas)", "Pembeli (Analisis Iklan Marketplace)")
)
is_penjual = "Penjual" in persona

daftar_merek = sorted(df['merek'].unique())

# ============================================================
# CABANG LOGIKA: PENJUAL (INPUT MANDIRI) VS PEMBELI (EXPLORE)
# ============================================================
if is_penjual:
    st.title("📝 Formulir Uji Kelayakan Iklan Mandiri (Khusus Penjual)")
    st.write("Masukkan spesifikasi mobil dan ketik rencana teks deskripsi iklan Anda. Mesin AI akan menganalisis konten secara *real-time* untuk memprediksi harga wajar dan kecepatan serap pasar.")
    
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        merek_dipilih = st.selectbox("Pilih Merek Mobil:", daftar_merek)
        df_model_merek = df[df['merek'] == merek_dipilih]
        daftar_model = sorted(df_model_merek['model'].unique())
        model_dipilih = st.selectbox("Pilih Model/Seri:", daftar_model)
        tahun_dipilih = st.number_input("Tahun Perakitan Mobil:", min_value=1995, max_value=2026, value=2018)
        transmisi_dipilih = st.selectbox("Jenis Transmisi:", ["Automatic", "Manual"])

    with col_in2:
        km_dipilih = st.number_input("Jarak Tempuh (KM di Odometer):", min_value=100, max_value=500000, value=65000, step=5000)
        harga_dipilih = st.number_input("Rencana Harga Jual Iklan (Rp):", min_value=15000000, max_value=3000000000, value=150000000, step=5000000)
        tipe_penjual_dipilih = st.selectbox("Profil Penjual:", ["Individu", "Dealer"])

    deskripsi_teks = st.text_area(
        "Ketik Rencana Deskripsi Iklan Anda:",
        value="Mobil terawat mulus, service record resmi bengkel berkala, pemakaian pribadi, bebas banjir dan tidak pernah tabrakan. Siap pakai jarak jauh.",
        height=100
    )

    # NLP Real-Time Scanning
    teks_lower = deskripsi_teks.lower()
    pola_garansi = r'\b(garansi|warranty|otospector|carsome|bebas banjir|bebas tabrak|service record|buku servis|terawat)\b'
    pola_kredit = r'\b(tdp|dp minim|angsuran|cicilan|over kredit|paket kredit)\b'
    
    ada_garansi_deteksi = "Ya" if re.search(pola_garansi, teks_lower) else "Tidak"
    indikasi_tdp_deteksi = 1 if re.search(pola_kredit, teks_lower) else 0

    st.caption(f"🔍 **Analisis AI Teks:** Fitur Nilai Tambah Terdeteksi = **{ada_garansi_deteksi}** | Skema Kredit Terdeteksi = **{'Ya' if indikasi_tdp_deteksi else 'Tidak'}**")

    # Evaluasi Mandiri
    harga_wajar_ml, deviasi_asli, status_asli, alasan_asli, is_jebakan_asli, usia_mobil, km_per_tahun = evaluasi_likuiditas(
        merek_dipilih, model_dipilih, tahun_dipilih, transmisi_dipilih, km_dipilih,
        tipe_penjual_dipilih, ada_garansi_deteksi, harga_dipilih, indikasi_tdp_deteksi
    )
    unit_label = f"{merek_dipilih} {model_dipilih} ({tahun_dipilih})"

else:
    # Mode Pembeli (Pilih dari unit aktif)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filter Marketplace")
    merek_terpilih = st.sidebar.selectbox("Pilih Merek:", daftar_merek)

    df_merek = df[df['merek'] == merek_terpilih]
    daftar_model = sorted(df_merek['model'].unique())
    model_terpilih = st.sidebar.selectbox("Pilih Model/Seri:", daftar_model)

    df_unit_cocok = df_merek[df_merek['model'] == model_terpilih].reset_index(drop=True)
    if df_unit_cocok.empty:
        st.warning("Tidak ditemukan unit untuk model tersebut.")
        st.stop()

    opsi_unit = [
        f"No.{i+1} | Thn {int(r['tahun'])} | {str(r['transmisi']).title()} | Rp {int(r['harga']):,} | {int(r['jarak_tempuh']):,} KM"
        for i, r in df_unit_cocok.iterrows()
    ]
    indeks_terpilih = st.sidebar.selectbox("Pilih Unit dari Marketplace:", range(len(opsi_unit)), format_func=lambda x: opsi_unit[x])
    unit = df_unit_cocok.iloc[indeks_terpilih]

    harga_wajar_ml, deviasi_asli, status_asli, alasan_asli, is_jebakan_asli, usia_mobil, km_per_tahun = evaluasi_likuiditas(
        unit['merek'], unit['model'], int(unit['tahun']), str(unit['transmisi']), int(unit['jarak_tempuh']),
        str(unit.get('tipe_penjual', 'Individu')), str(unit.get('ada_garansi', 'Tidak')), float(unit['harga']), int(unit.get('indikasi_kredit_tdp', 0))
    )
    harga_dipilih = float(unit['harga'])
    unit_label = f"{unit['merek']} {unit['model']} ({int(unit['tahun'])}) — {unit.get('judul', '-')}"
    km_dipilih = int(unit['jarak_tempuh'])
    tipe_penjual_dipilih = str(unit.get('tipe_penjual', 'Individu'))
    merek_dipilih = unit['merek']
    model_dipilih = unit['model']
    tahun_dipilih = int(unit['tahun'])

    st.title("📊 AutoLiquid: Evaluasi Deal Marketplace (Khusus Pembeli)")

st.markdown("---")

# ============================================================
# DASHBOARD HASIL VALUASI
# ============================================================
st.markdown(f"### Analisis Unit: `{unit_label}`")

if is_jebakan_asli:
    st.error("⚠️ **WASPADA INDIKASI HARGA JEBAKAN (TDP/KREDIT):** Harga penawaran terindikasi sebagai skema uang muka (DP), bukan pelunasan tunai keseluruhan unit mobil.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Harga Penawaran", f"Rp {int(harga_dipilih):,}")
col1.caption(f"Profil: {tipe_penjual_dipilih}")

col2.metric("Estimasi Harga Wajar (ML)", f"Rp {int(harga_wajar_ml):,}")
col2.caption("Nilai Pasar Wajar Standar")

col3.metric("Deviasi terhadap Pasar", f"{deviasi_asli:+.1f}%")
col3.caption("Taraf daya saing harga")

col4.metric("Odometer", f"{km_dipilih:,} KM")
col4.caption(f"Rata-rata: {km_per_tahun:,} KM/thn")

st.markdown("---")

col_kiri, col_kanan = st.columns([1.2, 1.8])

with col_kiri:
    st.subheader("Vonis Likuiditas Penjualan")
    if status_asli == "Cepat":
        st.success("### [✓] CEPAT LAKU (< 15 Hari)")
    elif status_asli == "Sedang":
        st.warning("### [~] LAKU SEDANG (15 - 30 Hari)")
    else:
        st.error("### [!] LAMBAT LAKU (> 30 Hari / Mengendap)")
    st.write(f"**Diagnosa Sistem:** {alasan_asli}")

with col_kanan:
    st.subheader(f"Rekomendasi Keputusan ({'Penjual' if is_penjual else 'Pembeli'})")
    target_cepat = harga_wajar_ml * 0.95
    target_sedang = harga_wajar_ml * 1.00
    batas_nego = harga_wajar_ml * 0.92

    if is_penjual:
        if status_asli == "Cepat":
            st.info("Harga iklan Anda sangat atraktif dan berada dalam zona penyerapan cepat. Potensi unit terjual dalam hitungan hari sangat tinggi.")
        elif status_asli == "Sedang":
            st.info(f"Harga berada dalam rentang wajar. Jika butuh dana cepat, sesuaikan harga ke sekitar **Rp {int(target_cepat):,}**.")
        else:
            st.info(f"Harga tergolong kemahalan. Disarankan mengoreksi harga ke **Rp {int(target_sedang):,}** (Pasar Standar) atau **Rp {int(target_cepat):,}** (Cepat Laku).")
    else:
        if status_asli == "Cepat":
            st.info(f"Unit ini terindikasi **Fair Deal / Best Offer**. Segera jadwalkan survei fisik kendaraan. Batas negosiasi agresif berada di kisaran **Rp {int(batas_nego):,}**.")
        elif status_asli == "Sedang":
            st.info(f"Harga sesuai dengan patokan rata-rata pasar. Lakukan negosiasi wajar dengan target akhir di kisaran **Rp {int(target_cepat):,}**.")
        else:
            st.info(f"Unit terindikasi **Overpriced**. Jangan membeli di harga iklan; gunakan nilai wajar **Rp {int(harga_wajar_ml):,}** sebagai patokan tawar.")

st.markdown("---")

# ============================================================
# PRICE DROP SIMULATOR
# ============================================================
st.subheader("💡 Simulator Penyesuaian Harga Dinamis (Price Drop Simulator)")
st.caption("Uji perubahan harga jual menggunakan slider untuk melihat bagaimana respon pasar berubah seketika.")

slider_min = max(10_000_000, int(harga_wajar_ml * 0.60))
slider_max = int(harga_wajar_ml * 1.40)

harga_sim = st.slider(
    "Simulasikan Rencana Harga Baru (Rp):",
    min_value=slider_min,
    max_value=slider_max,
    value=int(max(slider_min, min(slider_max, harga_dipilih))),
    step=1_000_000,
    format="Rp %d"
)

dev_sim = ((harga_sim - harga_wajar_ml) / harga_wajar_ml) * 100
if dev_sim <= -3.0:
    stat_sim = "Cepat"
elif -3.0 < dev_sim <= 7.0:
    stat_sim = "Sedang"
else:
    stat_sim = "Lambat"

col_s1, col_s2, col_s3 = st.columns(3)
col_s1.metric("Harga Simulasi", f"Rp {harga_sim:,}", delta=f"{harga_sim - harga_dipilih:+,}")
col_s2.metric("Deviasi Pasar Baru", f"{dev_sim:+.1f}%")
col_s3.metric("Status Likuiditas Baru", stat_sim)

if stat_sim == "Cepat":
    st.success(f"Pada nominal **Rp {harga_sim:,}**, unit langsung masuk ke zona **Cepat Laku** (< 15 hari).")
elif stat_sim == "Sedang":
    st.warning(f"Pada nominal **Rp {harga_sim:,}**, unit berada di zona **Laku Sedang** (15 - 30 hari).")
else:
    st.error(f"Pada nominal **Rp {harga_sim:,}**, unit berada di zona **Lambat Laku** (> 30 hari).")