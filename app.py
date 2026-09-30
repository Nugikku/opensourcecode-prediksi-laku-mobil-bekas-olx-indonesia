"""
AutoLaku: Sistem Prediksi Kecepatan Laku Mobil Bekas OLX (antarmuka Streamlit).
Logika prediksi SEPENUHNYA ada di mesin_prediksi.py (sama dgn yg dipakai
test_prediksi_klasifikasi.py & test_sistem.py) -- app.py ini murni tampilan,
tidak ada aturan if-else manual di sini.
"""

import streamlit as st
import pandas as pd

from mesin_prediksi import muat_sistem, daftar_pilihan, prediksi, InputTidakValid
from processing import AMBANG_KEYAKINAN_SAMPEL

st.set_page_config(page_title="AutoLaku - Prediksi Kecepatan Laku Mobil Bekas OLX",
                    page_icon="🚗", layout="wide")

# ============================================================
# MUAT SISTEM (model + dataset + meta) -- sekali saja, di-cache
# ============================================================
@st.cache_resource
def _muat():
    return muat_sistem()

try:
    sistem = _muat()
except FileNotFoundError as e:
    st.error(f"{e}")
    st.stop()

df_pilihan = daftar_pilihan(sistem)

st.title("🚗 AutoLaku")
st.caption("Sistem Prediksi Kecepatan Laku Mobil Bekas OLX Berdasarkan Harga Wajar, "
           "Jarak Tempuh, dan Deskripsi Iklan Menggunakan Metode Klasifikasi.")
st.info("Sistem ini untuk **penjual**: masukkan spesifikasi mobil dan rencana harga jual, "
        "sistem memprediksi apakah unit akan Cepat, Sedang, atau Lambat laku.")

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
    help="Dipakai sistem untuk mendeteksi ada/tidaknya garansi & indikasi kredit/TDP (termasuk negasi, mis. 'tidak ada garansi')."
)

jalankan = st.button("🔍 Prediksi Kecepatan Laku", type="primary")

if not jalankan:
    st.stop()

# ============================================================
# JALANKAN PREDIKSI (lewat mesin_prediksi.py, bukan rule manual)
# ============================================================
try:
    hasil = prediksi(sistem, merek, model, int(tahun), transmisi, int(km), float(harga_jual),
                     deskripsi, tipe_penjual=tipe_penjual)
except InputTidakValid as e:
    st.error(f"Input tidak valid: {e}")
    st.stop()

st.markdown("---")
st.subheader(f"📊 Hasil Prediksi: {merek} {model} ({hasil['tahun']})")

for p in hasil['peringatan']:
    st.warning(f"⚠️ {p}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rencana Harga Jual", f"Rp {int(hasil['harga_iklan']):,}")
c2.metric("Estimasi Harga Wajar", f"Rp {int(hasil['harga_wajar']):,}", help=hasil['sumber_harga_wajar'])
c3.metric("Deviasi Harga", f"{hasil['deviasi_persen']:+.1f}%")
c4.metric("Odometer", f"{hasil['km']:,} KM", help=f"{hasil['km_per_tahun']:,} KM/tahun")

st.caption(f"Garansi/Riwayat terdeteksi: **{hasil['ada_garansi']}** | "
           f"Indikasi kredit/TDP: **{'Ya' if hasil['indikasi_kredit_tdp'] else 'Tidak'}** | "
           f"Data pembanding: **{hasil['jumlah_sampel']} unit sejenis** (ambang keyakinan: {AMBANG_KEYAKINAN_SAMPEL})")

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
        st.caption("⚠️ Keyakinan prediksi RENDAH untuk kombinasi ini — anggap sebagai referensi kasar.")

with col_kanan:
    st.subheader("Peluang Tiap Kelas")
    if hasil['proba']:
        df_proba = pd.DataFrame({
            "Kelas": list(hasil['proba'].keys()),
            "Peluang": [v * 100 for v in hasil['proba'].values()],
        }).set_index("Kelas").reindex(["Cepat", "Sedang", "Lambat"])
        st.bar_chart(df_proba)
    else:
        st.caption("Model classifier ini tidak menyediakan skor peluang per kelas.")

# ============================================================
# SIMULATOR PENYESUAIAN HARGA
# ============================================================
st.markdown("---")
st.subheader("💡 Simulator Penyesuaian Harga")
st.caption("Geser slider untuk melihat bagaimana prediksi berubah pada rencana harga jual yang berbeda "
           "(diprediksi ulang lewat model yang sama, bukan rumus terpisah).")

hw = hasil['harga_wajar']
slider_min = max(10_000_000, int(hw * 0.5))
slider_max = int(hw * 2.0)
nilai_awal = int(min(max(hasil['harga_iklan'], slider_min), slider_max))

harga_sim = st.slider("Rencana Harga Jual Simulasi (Rp)", min_value=slider_min, max_value=slider_max,
                      value=nilai_awal, step=1_000_000, format="Rp %d")

hasil_sim = prediksi(sistem, merek, model, int(tahun), transmisi, int(km), float(harga_sim), deskripsi,
                     tipe_penjual=tipe_penjual)

cs1, cs2, cs3 = st.columns(3)
cs1.metric("Harga Simulasi", f"Rp {harga_sim:,}", delta=f"{harga_sim - int(hasil['harga_iklan']):+,}")
cs2.metric("Deviasi Baru", f"{hasil_sim['deviasi_persen']:+.1f}%")
cs3.metric("Prediksi Baru", hasil_sim['status'])

if hasil_sim['status'] == "Cepat":
    st.success(f"Pada Rp {harga_sim:,}, prediksi sistem: **Cepat Laku**.")
elif hasil_sim['status'] == "Sedang":
    st.warning(f"Pada Rp {harga_sim:,}, prediksi sistem: **Sedang Laku**.")
else:
    st.error(f"Pada Rp {harga_sim:,}, prediksi sistem: **Lambat Laku**.")

st.markdown("---")
st.caption("Label Cepat/Sedang/Lambat bersifat proxy (dibentuk dari deviasi harga terhadap harga wajar), "
           "bukan data status terjual historis sungguhan — karena OLX tidak menyediakannya. "
           "Gunakan sebagai referensi, bukan kepastian.")