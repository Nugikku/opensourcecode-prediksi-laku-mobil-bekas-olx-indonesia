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

# ============================================================
# SIDEBAR METADATA ML (P3-4, P5, P6, P12)
# ============================================================
meta = sistem.get("meta", {})
with st.sidebar:
    st.header("⚙️ Informasi Sistem ML")
    st.markdown(f"**Model Klasifikasi:** `{meta.get('model_klasifikasi', 'Random Forest')}`")
    st.markdown(f"**Model Regresi Harga:** `{meta.get('model_regresi', 'Ridge')}`")
    if 'mape_harga_wajar_pct' in meta:
        st.markdown(f"**MAPE Harga Wajar:** `{meta['mape_harga_wajar_pct']}%`")
    if 'f1_macro_test' in meta:
        st.markdown(f"**F1-Score Test:** `{meta['f1_macro_test']*100:.2f}%`")
    if 'accuracy_test' in meta:
        st.markdown(f"**Akurasi Test:** `{meta['accuracy_test']*100:.2f}%`")
    if 'q33' in meta and 'q67' in meta:
        st.markdown(f"**Ambang Tertile (Hanya dari Train):**  \n- Cepat: $\\le {meta['q33']:.2f}\\%$  \n- Sedang: ${meta['q33']:.2f}\\%$ s/d ${meta['q67']:.2f}\\%$  \n- Lambat: $> {meta['q67']:.2f}\\%$")
    st.markdown("---")
    st.caption("🛡️ **Bebas Data Leakage (P12):** Ambang tertile & model regresi dihitung murni dari data training (80%) sebelum label proxy dibuat.")

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

deskripsi = st.text_area(
    "Rencana Deskripsi Iklan",
    value="Mobil terawat, pajak hidup, service record resmi, bebas banjir dan tidak pernah tabrakan.",
    height=90,
    help="Dipakai sistem untuk mendeteksi ada/tidaknya garansi & indikasi kredit/TDP (termasuk negasi, mis. 'tidak ada garansi')."
)

# Inisialisasi state untuk persistensi tampilan saat slider digeser
if "sudah_prediksi" not in st.session_state:
    st.session_state.sudah_prediksi = False

jalankan = st.button("🔍 Prediksi Kecepatan Laku", type="primary")

if jalankan:
    st.session_state.sudah_prediksi = True
    st.session_state.input_aktif = {
        "merek": merek,
        "model": model,
        "tahun": int(tahun),
        "transmisi": transmisi,
        "km": int(km),
        "harga_jual": float(harga_jual),
        "deskripsi": deskripsi,
    }
    st.session_state.reset_slider = True

if not st.session_state.get("sudah_prediksi") or "input_aktif" not in st.session_state:
    st.stop()

inp = st.session_state.input_aktif

# ============================================================
# JALANKAN PREDIKSI (lewat mesin_prediksi.py, bukan rule manual)
# ============================================================
try:
    hasil = prediksi(sistem, inp['merek'], inp['model'], inp['tahun'], inp['transmisi'],
                     inp['km'], inp['harga_jual'], inp['deskripsi'])
except InputTidakValid as e:
    st.error(f"Input tidak valid: {e}")
    st.session_state.sudah_prediksi = False
    st.stop()

st.markdown("---")
st.subheader(f"📊 Hasil Prediksi: {inp['merek']} {inp['model']} ({hasil['tahun']})")

for p in hasil['peringatan']:
    st.warning(f"⚠️ {p}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rencana Harga Jual", f"Rp {int(hasil['harga_iklan']):,}")
c2.metric("Estimasi Harga Wajar", f"Rp {int(hasil['harga_wajar']):,}", help=hasil['sumber_harga_wajar'])
c3.metric("Deviasi Harga", f"{hasil['deviasi_persen']:+.1f}%")
c4.metric("Odometer", f"{hasil['km']:,} KM", help=f"{hasil['km_per_tahun']:,} KM/tahun")

st.markdown("##### 🔍 Fitur Teks & Kondisi Terdeteksi dari Deskripsi")
tf1, tf2, tf3, tf4, tf5 = st.columns(5)
tf1.metric("Garansi", hasil['ada_garansi'])
tf2.metric("Pajak Hidup", hasil['pajak_hidup'])
tf3.metric("Catatan Servis", hasil['catatan_servis'])
tf4.metric("Riwayat Negatif", hasil['riwayat_negatif'])
tf5.metric("Indikasi Kredit/TDP", "Ya" if hasil['indikasi_kredit_tdp'] else "Tidak")

st.caption(f"Data pembanding: **{hasil['jumlah_sampel']} unit sejenis** (ambang batas keyakinan: {AMBANG_KEYAKINAN_SAMPEL})")

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
slider_max = max(slider_min + 10_000_000, int(hw * 2.0))
nilai_awal = int(min(max(hasil['harga_iklan'], slider_min), slider_max))

# Reset slider jika tombol prediksi baru ditekan atau nilai di luar batas min/max
if st.session_state.get("reset_slider", False) or "slider_harga_sim" not in st.session_state:
    st.session_state.slider_harga_sim = nilai_awal
    st.session_state.reset_slider = False
elif st.session_state.slider_harga_sim < slider_min or st.session_state.slider_harga_sim > slider_max:
    st.session_state.slider_harga_sim = nilai_awal

harga_sim = st.slider(
    "Rencana Harga Jual Simulasi (Rp)",
    min_value=slider_min,
    max_value=slider_max,
    step=1_000_000,
    format="Rp %d",
    key="slider_harga_sim"
)

hasil_sim = prediksi(sistem, inp['merek'], inp['model'], inp['tahun'], inp['transmisi'],
                     inp['km'], float(harga_sim), inp['deskripsi'])

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