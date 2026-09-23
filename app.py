import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="AutoValuate - Decision Support System Likuiditas Mobil",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

FILE_MODEL = "model_klasifikasi_penjualan.pkl"
FILE_DATA = "dataset_olx_bersih.csv"

# ============================================================
# MEMUAT MODEL DAN DATASET
# ============================================================
@st.cache_resource
def load_model():
    if not os.path.exists(FILE_MODEL):
        return None
    with open(FILE_MODEL, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_dataset():
    if not os.path.exists(FILE_DATA):
        return None
    df = pd.read_csv(FILE_DATA)
    # Filter unit yang terisi lengkap
    df_valid = df[
        (df['harga'].notna()) & 
        (df['tahun'].notna()) & 
        (df['merek'].notna()) &
        (df['merek'].str.strip() != '') &
        (df['merek'].str.lower() != 'lainnya')
    ].copy().reset_index(drop=True)
    return df_valid

model_pipeline = load_model()
df = load_dataset()

if model_pipeline is None or df is None:
    st.error("File model (.pkl) atau dataset (.csv) tidak ditemukan. Pastikan proses processing dan training sudah dijalankan.")
    st.stop()

# ============================================================
# SIDEBAR: SELEKSI UNIT & PERSONA
# ============================================================
st.sidebar.title("🚗 AutoValuate DSS")
st.sidebar.caption("Sistem Pendukung Keputusan Likuiditas Mobil Bekas")

persona = st.sidebar.radio(
    "Sudut Pandang Pengguna:",
    ("Penjual (Strategi Penjualan)", "Pembeli (Analisis Deal)")
)
is_penjual = "Penjual" in persona

st.sidebar.markdown("---")
st.sidebar.subheader("Filter Katalog Kendaraan")

daftar_merek = sorted(df['merek'].unique())
merek_dipilih = st.sidebar.selectbox("Pilih Merek Kendaraan:", daftar_merek)

df_merek = df[df['merek'] == merek_dipilih]
daftar_model = sorted(df_merek['model'].dropna().unique())
model_dipilih = st.sidebar.selectbox("Pilih Model/Seri:", daftar_model)

df_kandidat = df_merek[df_merek['model'] == model_dipilih].reset_index(drop=True)

if df_kandidat.empty:
    st.warning("Tidak ada unit yang cocok dengan filter yang dipilih.")
    st.stop()

# Pilihan unit spesifik dari daftar
opsi_unit = [
    f"No.{i+1} | {r['tahun']} | {str(r['transmisi']).title()} | Rp {int(r['harga']):,} | {int(r['jarak_tempuh']):,} KM"
    for i, r in df_kandidat.iterrows()
]
indeks_unit = st.sidebar.selectbox("Pilih Unit Riil dari Marketplace:", range(len(opsi_unit)), format_func=lambda x: opsi_unit[x])
unit = df_kandidat.iloc[indeks_unit]

# ============================================================
# AREA UTAMA: DETAIL UNIT TERPILIH
# ============================================================
st.title("📊 Evaluasi Pasar & Prediksi Likuiditas")
st.markdown(f"**Unit:** `{unit['merek']} {unit['model']} ({int(unit['tahun'])})` — {str(unit.get('judul', '-'))}")

col_info1, col_info2, col_info3, col_info4 = st.columns(4)
col_info1.metric("Harga Penawaran Iklan", f"Rp {int(unit['harga']):,}")
col_info2.metric("Odometer", f"{int(unit['jarak_tempuh']):,} KM")
col_info3.metric("Lama Tayang Riil", f"{int(unit.get('durasi_tayang_hari', 0))} Hari")
col_info4.metric("Status Data Riil", str(unit.get('kategori_penjualan', '-')))

# Ekstraksi fitur pendukung
tahun_unit = int(unit['tahun'])
usia_mobil = int(unit.get('usia_mobil', max(1, 2026 - tahun_unit)))
km_per_tahun = int(unit.get('km_per_tahun', round(int(unit['jarak_tempuh']) / usia_mobil)))
tipe_penjual = str(unit.get('tipe_penjual', 'Individu'))
ada_garansi = str(unit.get('ada_garansi', 'Tidak'))
harga_asli = float(unit['harga'])

df_lokal = df[(df['model'] == unit['model']) & (df['tahun'] == tahun_unit)]
median_pasar = df_lokal['harga'].median() if not df_lokal.empty else harga_asli
deviasi_asli = ((harga_asli - median_pasar) / median_pasar) * 100

st.markdown("---")

# ============================================================
# PREDIKSI MODEL PADA DATA SAAT INI
# ============================================================
def evaluasi_unit(harga_tes, deviasi_tes):
    data_row = pd.DataFrame([{
        'merek': unit['merek'],
        'model': unit['model'],
        'transmisi': str(unit['transmisi']).lower(),
        'tipe_penjual': tipe_penjual,
        'ada_garansi': ada_garansi,
        'harga': harga_tes,
        'tahun': tahun_unit,
        'jarak_tempuh': int(unit['jarak_tempuh']),
        'usia_mobil': usia_mobil,
        'km_per_tahun': km_per_tahun,
        'deviasi_harga_pasar': deviasi_tes
    }])
    pred_kelas = model_pipeline.predict(data_row)[0]
    prob_dict = {}
    if hasattr(model_pipeline, "predict_proba"):
        probs = model_pipeline.predict_proba(data_row)[0]
        for k, p in zip(model_pipeline.classes_, probs):
            prob_dict[k] = p * 100
    return pred_kelas, prob_dict

prediksi_asli, prob_asli = evaluasi_unit(harga_asli, deviasi_asli)

col_pred, col_diag = st.columns([1.2, 2])

with col_pred:
    st.subheader("Hasil Prediksi Model")
    if prediksi_asli == 'Cepat':
        st.success(f"### [✓] CEPAT LAKU (< 15 Hari)")
    elif prediksi_asli == 'Sedang':
        st.warning(f"### [~] LAKU SEDANG (15 - 30 Hari)")
    else:
        st.error(f"### [!] LAMBAT LAKU (> 30 Hari)")
    
    if prob_asli:
        st.write("**Distribusi Keyakinan Multi-Class:**")
        for kls in ['Cepat', 'Sedang', 'Lambat']:
            val = prob_asli.get(kls, 0.0)
            st.progress(int(val), text=f"{kls}: {val:.1f}%")

with col_diag:
    st.subheader("Diagnosa Pasar")
    st.write(f"- **Median Harga Pasar (Seri & Tahun Sama):** Rp {int(median_pasar):,}")
    st.write(f"- **Deviasi terhadap Pasar:** `{deviasi_asli:+.1f}%`")
    st.write(f"- **Profil Penjual:** `{tipe_penjual}` | **Jaminan Garansi:** `{ada_garansi}`")
    st.write(f"- **Intensitas Pemakaian:** `{km_per_tahun:,} KM/tahun`")

    st.markdown("##### Kesimpulan Keputusan:")
    if is_penjual:
        if prediksi_asli == 'Cepat':
            st.info("Penetapan harga Anda berada di zona penyerapan cepat. Iklan berpeluang besar terjual dalam waktu singkat tanpa perlu kompromi harga.")
        elif prediksi_asli == 'Sedang':
            st.info("Harga iklan berada di rentang wajar namun kompetitif. Diperlukan waktu penyerapan normal di pasar.")
        else:
            st.info("Harga terindikasi terlalu tinggi dibandingkan kombinasi jarak tempuh dan usia unit. Unit berisiko tertahan lama di marketplace.")
    else:
        if prediksi_asli == 'Cepat':
            st.info("Unit merupakan **Fair Deal / Best Price**. Kondisi dan penawaran harga sangat menguntungkan pembeli. Prioritaskan inspeksi fisik secepatnya.")
        elif prediksi_asli == 'Sedang':
            st.info("Harga berada pada batas pasar standar. Pembeli disarankan melakukan negosiasi wajar.")
        else:
            st.info("Unit terindikasi **Overpriced**. Hindari transaksi sebelum penjual bersedia menurunkan harga mendekati rata-rata pasar.")

st.markdown("---")

# ============================================================
# INOVASI: PRICE DROP SIMULATOR (SIMULATOR INTERAKTIF)
# ============================================================
st.subheader("💡 Simulasi Penyesuaian Harga (Price Simulator)")
st.caption("Uji perubahan harga penawaran menggunakan slider di bawah untuk melihat bagaimana model mengubah status likuiditas secara real-time.")

slider_min = int(harga_asli * 0.7)
slider_max = int(harga_asli * 1.3)
harga_simulasi = st.slider(
    "Geser untuk mensimulasikan harga baru (Rp):",
    min_value=slider_min,
    max_value=slider_max,
    value=int(harga_asli),
    step=1_000_000,
    format="Rp %d"
)

deviasi_simulasi = ((harga_simulasi - median_pasar) / median_pasar) * 100
pred_simulasi, prob_simulasi = evaluasi_unit(harga_simulasi, deviasi_simulasi)

col_sim1, col_sim2, col_sim3 = st.columns(3)
col_sim1.metric("Harga Simulasi", f"Rp {harga_simulasi:,}", delta=f"{harga_simulasi - harga_asli:+,}")
col_sim2.metric("Deviasi Pasar Baru", f"{deviasi_simulasi:+.1f}%")
col_sim3.metric("Status Likuiditas Baru", pred_simulasi)

if pred_simulasi == 'Cepat':
    st.success(f"Pada nominal **Rp {harga_simulasi:,}**, unit diproyeksikan langsung masuk ke kategori **Cepat Laku** (< 15 hari).")
elif pred_simulasi == 'Sedang':
    st.warning(f"Pada nominal **Rp {harga_simulasi:,}**, unit diproyeksikan berada di kategori **Laku Sedang** (15 - 30 hari).")
else:
    st.error(f"Pada nominal **Rp {harga_simulasi:,}**, unit tetap berada di kategori **Lambat Laku** (> 30 hari).")