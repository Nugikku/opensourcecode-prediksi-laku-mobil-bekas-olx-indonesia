"""
app.py
--------------------------------------------------------------------------------
AutoLaku: Sistem Prediksi Kecepatan Laku Mobil Bekas OLX (Antarmuka Streamlit)
--------------------------------------------------------------------------------
Logika prediksi sepenuhnya dikelola oleh mesin_prediksi.py.
Script ini fokus murni pada antarmuka pengguna (UI/UX).
"""

import streamlit as st
import pandas as pd

from mesin_prediksi import muat_sistem, daftar_pilihan, prediksi, InputTidakValid
from processing import AMBANG_KEYAKINAN_SAMPEL

# Konfigurasi Halaman
st.set_page_config(
    page_title="AutoLaku - Prediksi Kecepatan Laku Mobil Bekas OLX",
    page_icon="🚗",
    layout="wide"
)

# Helper Function Format Rupiah (Standar Indonesia)
def format_rupiah(nominal):
    return f"Rp {int(nominal):,}".replace(",", ".")

# ============================================================
# MUAT SISTEM (Model + Dataset + Metadata)
# ============================================================
@st.cache_resource
def _muat():
    return muat_sistem()

try:
    sistem = _muat()
except FileNotFoundError as e:
    st.error(f"Sistem gagal dimuat: {e}")
    st.stop()

df_pilihan = daftar_pilihan(sistem)

# Header Utama
st.title("🚗 AutoLaku")
st.caption("Sistem Prediksi Kecepatan Laku Mobil Bekas OLX Berdasarkan Harga Wajar, "
           "Jarak Tempuh, dan Deskripsi Iklan Menggunakan Metode Klasifikasi Machine Learning.")

st.info("💡 **Petunjuk Penjual:** Masukkan spesifikasi mobil dan rencana harga jual Anda. "
        "Sistem akan memprediksi estimasi kecepatan laku unit (**Cepat**, **Sedang**, atau **Lambat**).")

# ============================================================
# FORMULIR INPUT PENJUAL
# ============================================================
st.markdown("---")
st.subheader("📝 Formulir Input Penjual")

col1, col2 = st.columns(2)
with col1:
    merek = st.selectbox("Merek Mobil", sorted(df_pilihan['merek'].unique()))
    daftar_model = sorted(df_pilihan[df_pilihan['merek'] == merek]['model'].unique())
    model = st.selectbox("Model / Seri", daftar_model)
    tahun = st.number_input("Tahun Perakitan", min_value=1995, max_value=2026, value=2018, step=1)
    transmisi = st.selectbox("Transmisi", ["Automatic", "Manual"])

with col2:
    km = st.number_input("Jarak Tempuh (KM)", min_value=0, max_value=1_500_000, value=65_000, step=5_000)
    harga_jual = st.number_input("Rencana Harga Jual (Rp)", min_value=0, max_value=10_000_000_000,
                                  value=150_000_000, step=5_000_000)
    tipe_penjual = st.selectbox("Profil Penjual", ["Individu", "Dealer"])

deskripsi = st.text_area(
    "Rencana Deskripsi Iklan",
    value="Mobil terawat, pajak hidup, service record resmi, bebas banjir dan tidak pernah tabrakan.",
    height=90,
    help="Sistem mendeteksi keberadaan garansi, riwayat servis, serta skema kredit/TDP dari deskripsi ini."
)

jalankan = st.button("🔍 Prediksi Kecepatan Laku", type="primary", use_container_width=True)

if not jalankan:
    st.stop()

# ============================================================
# JALANKAN PREDIKSI (Lewat mesin_prediksi.py)
# ============================================================
try:
    hasil = prediksi(sistem, merek, model, int(tahun), transmisi, int(km), float(harga_jual),
                     deskripsi, tipe_penjual=tipe_penjual)
except InputTidakValid as e:
    st.error(f"Input tidak valid: {e}")
    st.stop()

st.markdown("---")
st.subheader(f"📊 Hasil Prediksi: {merek} {model} ({hasil['tahun']})")

# Peringatan Sistem jika ada
for p in hasil['peringatan']:
    st.warning(f"⚠️ {p}")

# Ringkasan Metrik
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rencana Harga Jual", format_rupiah(hasil['harga_iklan']))
c2.metric("Estimasi Harga Wajar", format_rupiah(hasil['harga_wajar']), help=hasil['sumber_harga_wajar'])
c3.metric("Deviasi Harga", f"{hasil['deviasi_persen']:+.1f}%")
c4.metric("Odometer", f"{hasil['km']:,}".replace(",", ".") + " KM", help=f"{hasil['km_per_tahun']:,}".replace(",", ".") + " KM/tahun")

st.caption(f"Garansi/Riwayat Terdeteksi: **{hasil['ada_garansi']}** | "
           f"Indikasi Kredit/TDP: **{'Ya' if hasil['indikasi_kredit_tdp'] else 'Tidak'}** | "
           f"Data Pembanding: **{hasil['jumlah_sampel']} unit sejenis** (Ambang Keyakinan: {AMBANG_KEYAKINAN_SAMPEL})")

st.markdown("---")
col_kiri, col_kanan = st.columns([1, 1.4])

with col_kiri:
    st.subheader("Prediksi Kecepatan Laku")
    status = hasil['status']
    if status == "Cepat":
        st.success("### ✅ CEPAT LAKU")
    elif status == "Sedang":
        st.warning("### 🟡 SEDANG LAKU")
    else:
        st.error("### 🔴 LAMBAT LAKU")

    if hasil['keyakinan_rendah']:
        st.caption("⚠️ Keyakinan prediksi rendah untuk kombinasi ini — gunakan sebagai referensi pertimbangan awal.")

with col_kanan:
    st.subheader("Peluang Tiap Kelas Likuiditas")
    if hasil['proba']:
        df_proba = pd.DataFrame({
            "Kelas": list(hasil['proba'].keys()),
            "Peluang (%)": [v * 100 for v in hasil['proba'].values()],
        }).set_index("Kelas").reindex(["Cepat", "Sedang", "Lambat"])
        st.bar_chart(df_proba)
    else:
        st.caption("Model klasifikasi ini tidak menyediakan skor probabilitas per kelas.")

# ============================================================
# SIMULATOR PENYESUAIAN HARGA (INTERAKTIF)
# ============================================================
st.markdown("---")
st.subheader("💡 Simulator Penyesuaian Harga Jual")
st.caption("Geser slider di bawah ini untuk menguji bagaimana perubahan harga jual mempengaruhi kecepatan laku kendaraan.")

hw = hasil['harga_wajar']
slider_min = max(10_000_000, int(hw * 0.5))
slider_max = int(hw * 2.0)
nilai_awal = int(min(max(hasil['harga_iklan'], slider_min), slider_max))

harga_sim = st.slider("Rencana Harga Jual Simulasi (Rp)", min_value=slider_min, max_value=slider_max,
                      value=nilai_awal, step=1_000_000)

hasil_sim = prediksi(sistem, merek, model, int(tahun), transmisi, int(km), float(harga_sim), deskripsi,
                     tipe_penjual=tipe_penjual)

cs1, cs2, cs3 = st.columns(3)
cs1.metric("Harga Simulasi", format_rupiah(harga_sim), delta=format_rupiah(harga_sim - int(hasil['harga_iklan'])))
cs2.metric("Deviasi Baru", f"{hasil_sim['deviasi_persen']:+.1f}%")
cs3.metric("Prediksi Baru", hasil_sim['status'])

if hasil_sim['status'] == "Cepat":
    st.success(f"Pada harga **{format_rupiah(harga_sim)}**, prediksi sistem: **CEPAT LAKU**")
elif hasil_sim['status'] == "Sedang":
    st.warning(f"Pada harga **{format_rupiah(harga_sim)}**, prediksi sistem: **SEDANG LAKU**")
else:
    st.error(f"Pada harga **{format_rupiah(harga_sim)}**, prediksi sistem: **LAMBAT LAKU**")

st.markdown("---")
st.caption("Catatan Metodologis: Label Kecepatan Laku (Cepat/Sedang/Lambat) dibentuk dari deviasi harga jual terhadap estimasi harga pasar wajar "
           "menggunakan distribusi persentil tertile data training. Gunakan sebagai estimasi pendukung keputusan.")