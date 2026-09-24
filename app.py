import os
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
# SIDEBAR: SELEKSI PERSONA & FILTER KENDARAAN
# ============================================================
st.sidebar.title("🚗 AutoLiquid DSS")
st.sidebar.caption("Hybrid Decision Support System: ML Regression + Liquidity Engine")

persona = st.sidebar.radio(
    "Sudut Pandang Pengguna:",
    ("Penjual (Strategi Harga & Likuiditas)", "Pembeli (Analisis Deal & Negosiasi)")
)
is_penjual = "Penjual" in persona

st.sidebar.markdown("---")
st.sidebar.subheader("Filter Marketplace")

daftar_merek = sorted(df['merek'].unique())
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

# ============================================================
# EKSTRAKSI ATRIBUT UNIT TERPILIH
# ============================================================
tahun_unit = int(unit['tahun'])
km_unit = int(unit['jarak_tempuh'])
usia_mobil = int(unit.get('usia_mobil', max(1, 2026 - tahun_unit)))
km_per_tahun = int(unit.get('km_per_tahun', round(km_unit / usia_mobil)))
transmisi_unit = str(unit['transmisi']).lower()
tipe_penjual = str(unit.get('tipe_penjual', 'Individu'))
ada_garansi = str(unit.get('ada_garansi', 'Tidak'))
harga_iklan = float(unit['harga'])
indikasi_tdp = int(unit.get('indikasi_kredit_tdp', 0))

# ============================================================
# ENGINE VALUASI & LIKUIDITAS DENGAN PENGAMAN MINUS
# ============================================================
def evaluasi_likuiditas(harga_input):
    data_tes = pd.DataFrame([{
        'merek': unit['merek'],
        'model': unit['model'],
        'transmisi': transmisi_unit,
        'tipe_penjual': tipe_penjual,
        'ada_garansi': ada_garansi,
        'tahun': tahun_unit,
        'jarak_tempuh': km_unit,
        'usia_mobil': usia_mobil,
        'km_per_tahun': km_per_tahun
    }])
    
    pred_mentah = float(model_pipeline.predict(data_tes)[0])
    BATAS_MINIMUM = 20_000_000.0

    # Pengaman ekstrapolasi linier untuk mobil tua
    if pred_mentah < BATAS_MINIMUM:
        df_sejenis = df[(df['merek'] == unit['merek']) & (df['model'] == unit['model'])]
        if not df_sejenis.empty:
            harga_wajar = max(BATAS_MINIMUM, float(df_sejenis['harga'].median() * 0.55))
        else:
            harga_wajar = BATAS_MINIMUM
    else:
        harga_wajar = pred_mentah

    deviasi = ((harga_input - harga_wajar) / harga_wajar) * 100
    is_jebakan = (deviasi <= -35.0 and indikasi_tdp == 1)

    # Status Likuiditas Dinamis
    if deviasi <= -3.0:
        if km_per_tahun > 30000 and ada_garansi == 'Tidak':
            status = "Sedang"
            alasan = "Harga di bawah pasar, namun kilometer tergolong tinggi tanpa jaminan garansi."
        else:
            status = "Cepat"
            alasan = "Harga penawaran sangat atraktif di bawah estimasi wajar pasar."
    elif -3.0 < deviasi <= 7.0:
        status = "Sedang"
        alasan = "Harga penawaran seimbang dengan nilai rata-rata pasar."
    else:
        status = "Lambat"
        alasan = "Harga penawaran melebihi harga wajar pasar (overpriced)."

    return harga_wajar, deviasi, status, alasan, is_jebakan

harga_wajar_ml, deviasi_asli, status_asli, alasan_asli, is_jebakan_asli = evaluasi_likuiditas(harga_iklan)

# ============================================================
# TAMPILAN UTAMA: DASHBOARD VALUASI
# ============================================================
st.title("📊 AutoLiquid: Evaluasi Valuasi & Likuiditas Pasar")
st.markdown(f"**Unit Analisis:** `{unit['merek']} {unit['model']} ({tahun_unit})` — *{unit.get('judul', '-')}*")

if is_jebakan_asli:
    st.error("⚠️ **WASPADA INDIKASI HARGA JEBAKAN (TDP/KREDIT):** Harga penawaran terindikasi sebagai skema uang muka (DP), bukan pelunasan tunai keseluruhan unit mobil.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Harga Iklan Marketplace", f"Rp {int(harga_iklan):,}")
col1.caption(f"Profil: {tipe_penjual}")

col2.metric("Estimasi Harga Wajar (ML)", f"Rp {int(harga_wajar_ml):,}")
col2.caption("Hasil Prediksi Machine Learning")

col3.metric("Deviasi terhadap Pasar", f"{deviasi_asli:+.1f}%")
col3.caption("Taraf kewajaran penawaran")

col4.metric("Odometer", f"{km_unit:,} KM")
col4.caption(f"Rata-rata: {km_per_tahun:,} KM/thn")

st.markdown("---")

# ============================================================
# HASIL KEPUTUSAN & REKOMENDASI PERSONA
# ============================================================
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
    st.subheader(f"Rekomendasi Strategis ({'Penjual' if is_penjual else 'Pembeli'})")
    target_cepat = harga_wajar_ml * 0.95
    target_sedang = harga_wajar_ml * 1.00
    batas_nego = harga_wajar_ml * 0.92

    if is_penjual:
        if status_asli == "Cepat":
            st.info("Harga iklan Anda sangat atraktif dan berada dalam zona penyerapan cepat. Iklan berpeluang besar terjual tanpa perlu diskon tambahan.")
        elif status_asli == "Sedang":
            st.info(f"Harga berada dalam batas wajar penyerapan pasar. Jika Anda butuh dana cepat, turunkan ke sekitar **Rp {int(target_cepat):,}**.")
        else:
            st.info(f"Harga tergolong terlalu tinggi dibanding kondisi pasar. Disarankan menurunkan harga ke **Rp {int(target_sedang):,}** (Standar) atau **Rp {int(target_cepat):,}** (Cepat Laku).")
    else:
        if status_asli == "Cepat":
            st.info(f"Unit ini merupakan **Fair Deal / Best Offer**. Segera jadwalkan survei fisik kendaraan. Batas negosiasi agresif berada di kisaran **Rp {int(batas_nego):,}**.")
        elif status_asli == "Sedang":
            st.info(f"Harga sesuai dengan patokan rata-rata pasar. Lakukan negosiasi wajar dengan target akhir di kisaran **Rp {int(target_cepat):,}**.")
        else:
            st.info(f"Unit terindikasi **Overpriced**. Jangan membeli di harga iklan; gunakan nilai estimasi wajar **Rp {int(harga_wajar_ml):,}** sebagai batas tawar tertinggi.")

st.markdown("---")

# ============================================================
# PRICE DROP SIMULATOR (SIMULATOR INTERAKTIF)
# ============================================================
st.subheader("💡 Simulator Penyesuaian Harga Dinamis (Price Drop Simulator)")
st.caption("Uji perubahan harga jual menggunakan slider untuk melihat bagaimana deviasi dan status likuiditas berubah seketika.")

slider_min = max(10_000_000, int(harga_wajar_ml * 0.60))
slider_max = int(harga_wajar_ml * 1.40)

harga_sim = st.slider(
    "Simulasikan Rencana Harga Baru (Rp):",
    min_value=slider_min,
    max_value=slider_max,
    value=int(max(slider_min, min(slider_max, harga_iklan))),
    step=1_000_000,
    format="Rp %d"
)

_, dev_sim, stat_sim, alasan_sim, _ = evaluasi_likuiditas(harga_sim)

col_s1, col_s2, col_s3 = st.columns(3)
col_s1.metric("Harga Simulasi", f"Rp {harga_sim:,}", delta=f"{harga_sim - harga_iklan:+,}")
col_s2.metric("Deviasi Pasar Baru", f"{dev_sim:+.1f}%")
col_s3.metric("Status Likuiditas Baru", stat_sim)

if stat_sim == "Cepat":
    st.success(f"Pada nominal **Rp {harga_sim:,}**, unit langsung masuk ke zona **Cepat Laku** (< 15 hari).")
elif stat_sim == "Sedang":
    st.warning(f"Pada nominal **Rp {harga_sim:,}**, unit berada di zona **Laku Sedang** (15 - 30 hari).")
else:
    st.error(f"Pada nominal **Rp {harga_sim:,}**, unit berada di zona **Lambat Laku** (> 30 hari).")