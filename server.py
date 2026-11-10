"""
server.py
--------------------------------------------------------------------------------
AutoLaku: Frontend & REST API Server (FastAPI + Jinja2 Templates + Tailwind CSS)
Terintegrasi langsung dengan Backend Terbaru (mesin_prediksi.py & processing.py)
--------------------------------------------------------------------------------
"""

import os
import re
import html
import json
import string
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from mesin_prediksi import (
    muat_sistem, daftar_pilihan, prediksi, InputTidakValid,
    TAHUN_MIN, TAHUN_SEKARANG, HARGA_MIN, HARGA_MAX, KM_MAX
)
from processing import (
    AMBANG_KEYAKINAN_SAMPEL, NEGASI, NEGASI_RIWAYAT,
    POLA_GARANSI, POLA_RIWAYAT_NEGATIF, POLA_CATATAN_SERVIS,
    POLA_PAJAK_HIDUP, POLA_KREDIT, ekstrak_fitur_teks
)

# Inisialisasi FastAPI
app = FastAPI(
    title="AutoLaku - Prediksi Likuiditas Mobil Bekas OLX",
    description="Sistem Prediksi Kecepatan Laku Mobil Bekas Berdasarkan Harga Wajar & NLP Deskripsi Iklan",
    version="2.1.0",
    debug=True
)

templates = Jinja2Templates(directory="templates")
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Daftar merek anomali (motor/truk/data tidak valid) yang dikecualikan dari tampilan web
MEREK_ANOMALI_DIKECUALIKAN = {
    "piaggio", "harley davidson", "kawasaki", "ktm", "dfsk",
    "ducati", "geely", "aion", "neta", "yamaha", "truk",
    "maxus", "volvo", "jaguar"
}

# Muat sistem AutoLaku dari backend terbaru
sistem = muat_sistem()
df_pilihan_raw = daftar_pilihan(sistem)
df_pilihan = df_pilihan_raw[
    ~df_pilihan_raw['merek'].fillna('').str.strip().str.lower().isin(MEREK_ANOMALI_DIKECUALIKAN)
].reset_index(drop=True)
meta_pelabelan = sistem.get("meta", {})
Q33_TERTILE = float(meta_pelabelan.get("q33", meta_pelabelan.get("q33_tertile", -7.2695)))
Q67_TERTILE = float(meta_pelabelan.get("q67", meta_pelabelan.get("q67_tertile", 6.5131)))


def sorot_nlp_teks(teks: str) -> Dict[str, Any]:
    """
    Memindai dan memberi tanda visual (highlight) pada kata kunci NLP sesuai processing.py terbaru:
    - Hijau: Garansi / Catatan Servis / Pajak Hidup / Bebas Banjir
    - Oranye: Skema Kredit / TDP
    - Merah: Riwayat Negatif (Banjir/Tabrak tanpa negasi) & Negasi
    """
    if not teks:
        return {"html": "", "tags": []}

    safe_teks = html.escape(teks)
    tags_terdeteksi = []

    pola_garansi_re = re.compile(POLA_GARANSI, re.IGNORECASE)
    pola_servis_re = re.compile(POLA_CATATAN_SERVIS, re.IGNORECASE)
    pola_pajak_re = re.compile(POLA_PAJAK_HIDUP, re.IGNORECASE)
    pola_riwayat_re = re.compile(POLA_RIWAYAT_NEGATIF, re.IGNORECASE)
    pola_kredit_re = re.compile(POLA_KREDIT, re.IGNORECASE)

    matches = []

    # 1. Deteksi garansi
    for m in pola_garansi_re.finditer(safe_teks):
        prefix = [w.lower().strip(string.punctuation) for w in safe_teks[:m.start()].split()[-2:]]
        negated = any(w in NEGASI for w in prefix)
        matches.append((m.start(), m.end(), m.group(), 'garansi', negated))
        tags_terdeteksi.append({
            "term": m.group(),
            "kategori": "Garansi",
            "negasi": negated,
            "tipe": "garansi"
        })

    # 2. Deteksi catatan servis
    for m in pola_servis_re.finditer(safe_teks):
        prefix = [w.lower().strip(string.punctuation) for w in safe_teks[:m.start()].split()[-2:]]
        negated = any(w in NEGASI for w in prefix)
        matches.append((m.start(), m.end(), m.group(), 'servis', negated))
        tags_terdeteksi.append({
            "term": m.group(),
            "kategori": "Catatan Servis",
            "negasi": negated,
            "tipe": "servis"
        })

    # 3. Deteksi pajak hidup
    for m in pola_pajak_re.finditer(safe_teks):
        prefix = [w.lower().strip(string.punctuation) for w in safe_teks[:m.start()].split()[-2:]]
        negated = any(w in NEGASI for w in prefix)
        matches.append((m.start(), m.end(), m.group(), 'pajak', negated))
        tags_terdeteksi.append({
            "term": m.group(),
            "kategori": "Pajak Hidup",
            "negasi": negated,
            "tipe": "pajak"
        })

    # 4. Deteksi riwayat negatif (banjir/tabrak/kecelakaan/bocor/keropos)
    for m in pola_riwayat_re.finditer(safe_teks):
        prefix = [w.lower().strip(string.punctuation) for w in safe_teks[:m.start()].split()[-2:]]
        negated = any(w in NEGASI_RIWAYAT for w in prefix)
        if not negated:
            matches.append((m.start(), m.end(), m.group(), 'buruk', False))
            tags_terdeteksi.append({
                "term": m.group(),
                "kategori": "Riwayat Negatif",
                "negasi": False,
                "tipe": "buruk"
            })
        else:
            matches.append((m.start(), m.end(), m.group(), 'garansi', False))

    # 5. Deteksi skema kredit / TDP
    for m in pola_kredit_re.finditer(safe_teks):
        prefix = [w.lower().strip(string.punctuation) for w in safe_teks[:m.start()].split()[-2:]]
        negated = any(w in NEGASI for w in prefix)
        matches.append((m.start(), m.end(), m.group(), 'kredit', negated))
        tags_terdeteksi.append({
            "term": m.group(),
            "kategori": "Skema Kredit / TDP",
            "negasi": negated,
            "tipe": "kredit"
        })

    # Hindari overlap rentang kata (sort by start ascending, pilih terpanjang)
    matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    clean_matches = []
    last_end = -1
    for m in matches:
        if m[0] >= last_end:
            clean_matches.append(m)
            last_end = m[1]

    # Replace dari belakang ke depan
    clean_matches.sort(key=lambda x: x[0], reverse=True)

    html_output = safe_teks
    for start, end, word, tipe, negated in clean_matches:
        if tipe == 'buruk':
            badge = f'<span class="bg-rose-100 text-rose-800 border border-rose-300 font-bold px-1.5 py-0.5 rounded text-xs" title="PERINGATAN: Indikasi Riwayat Negatif"><i class="fa-solid fa-triangle-exclamation text-[10px] mr-1"></i>{word} ⚠️</span>'
        elif negated:
            badge = f'<span class="bg-rose-100 text-rose-800 border border-rose-300 font-semibold px-1.5 py-0.5 rounded text-xs" title="Negasi terdeteksi"><i class="fa-solid fa-triangle-exclamation text-[10px] mr-1"></i>{word} (Negasi)</span>'
        elif tipe in ('garansi', 'servis'):
            badge = f'<span class="bg-emerald-100 text-emerald-800 border border-emerald-300 font-semibold px-1.5 py-0.5 rounded text-xs" title="Garansi / Catatan Servis">{word} <i class="fa-solid fa-check text-[10px] ml-0.5"></i></span>'
        elif tipe == 'pajak':
            badge = f'<span class="bg-amber-100 text-amber-800 border border-amber-300 font-semibold px-1.5 py-0.5 rounded text-xs" title="Pajak Hidup">{word} <i class="fa-solid fa-file-shield text-[10px] ml-0.5"></i></span>'
        else:  # kredit
            badge = f'<span class="bg-orange-100 text-orange-800 border border-orange-300 font-semibold px-1.5 py-0.5 rounded text-xs" title="Indikasi TDP / Pembiayaan">{word} <i class="fa-solid fa-credit-card text-[10px] ml-0.5"></i></span>'

        html_output = html_output[:start] + badge + html_output[end:]

    return {"html": html_output, "tags": tags_terdeteksi}


class PredictRequest(BaseModel):
    merek: str
    model: str
    tahun: int
    transmisi: str
    km: int
    harga_iklan: float
    deskripsi: Optional[str] = ""


class SimulateRequest(BaseModel):
    merek: str
    model: str
    tahun: int
    transmisi: str
    km: int
    harga_simulasi: float
    harga_awal: float
    deskripsi: Optional[str] = ""


@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    """Menampilkan halaman awal / landing page pengenalan sistem."""
    df_lengkap = sistem["df"]
    df_mobil_valid = df_lengkap[
        ~df_lengkap['merek'].fillna('').str.strip().str.lower().isin(MEREK_ANOMALI_DIKECUALIKAN)
    ]
    merek_counts = df_mobil_valid['merek'].value_counts()

    foto_meta_path = os.path.join("static", "images", "cars", "merek_foto.json")
    foto_data = {}
    if os.path.exists(foto_meta_path):
        try:
            with open(foto_meta_path, "r", encoding="utf-8") as f:
                foto_data = json.load(f)
        except Exception:
            pass

    merek_summary = []
    for k, v in merek_counts.items():
        if k.strip().lower() in MEREK_ANOMALI_DIKECUALIKAN:
            continue
        info = foto_data.get(k, {})
        slug = re.sub(r'[^a-zA-Z0-9]+', '_', k.lower()).strip('_')
        merek_summary.append({
            "merek": k,
            "total": int(v),
            "model_populer": info.get("model_populer", k),
            "foto_url": info.get("foto_url") or f"/static/images/cars/{slug}.jpg"
        })

    context = {
        "request": request,
        "total_data": len(df_mobil_valid),
        "total_merek": len(merek_summary),
        "merek_summary": merek_summary,
        "q33": Q33_TERTILE,
        "q67": Q67_TERTILE,
        "meta": meta_pelabelan,
        "regresi_eval": meta_pelabelan,
    }
    return templates.TemplateResponse(request=request, name="landing.html", context=context)


@app.get("/prediksi", response_class=HTMLResponse)
@app.get("/app", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
async def app_page(request: Request):
    """Menampilkan antarmuka kalkulator prediksi likuiditas & simulator (index.html)."""
    daftar_merek = sorted(df_pilihan['merek'].dropna().unique().tolist())
    default_merek = "Toyota" if "Toyota" in daftar_merek else daftar_merek[0]
    default_models = sorted(df_pilihan[df_pilihan['merek'] == default_merek]['model'].dropna().unique().tolist())

    df_lengkap = sistem["df"]
    df_mobil_valid = df_lengkap[
        ~df_lengkap['merek'].fillna('').str.strip().str.lower().isin(MEREK_ANOMALI_DIKECUALIKAN)
    ]

    context = {
        "request": request,
        "daftar_merek": daftar_merek,
        "default_merek": default_merek,
        "default_models": default_models,
        "tahun_min": TAHUN_MIN,
        "tahun_max": TAHUN_SEKARANG,
        "q33": Q33_TERTILE,
        "q67": Q67_TERTILE,
        "ambang_sampel": AMBANG_KEYAKINAN_SAMPEL,
        "meta": meta_pelabelan,
        "regresi_eval": meta_pelabelan,
        "total_data": len(df_mobil_valid)
    }
    return templates.TemplateResponse(request=request, name="index.html", context=context)


@app.get("/api/models")
async def get_models(merek: str = Query(...)):
    """Mengembalikan daftar model yang tersedia untuk merek tertentu."""
    sub = df_pilihan[df_pilihan['merek'].str.lower() == merek.lower()]
    models = sorted(sub['model'].dropna().unique().tolist())
    return {"merek": merek, "models": models}


@app.post("/api/predict")
async def api_predict(req: PredictRequest):
    """Menjalankan inferensi regresi harga wajar dan klasifikasi likuiditas lewat mesin_prediksi.py."""
    try:
        hasil = prediksi(
            sistem=sistem,
            merek=req.merek,
            model=req.model,
            tahun=req.tahun,
            transmisi=req.transmisi,
            km=req.km,
            harga_iklan=req.harga_iklan,
            deskripsi=req.deskripsi or ""
        )
    except InputTidakValid as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan inferensi: {str(e)}")

    hw = hasil['harga_wajar']
    batas_cepat = round(hw * (1.0 + Q33_TERTILE / 100.0))
    batas_lambat = round(hw * (1.0 + Q67_TERTILE / 100.0))

    nlp_scan = sorot_nlp_teks(req.deskripsi or "")

    peringatan = list(hasil['peringatan'])
    if hasil.get('riwayat_negatif') == 'Ya':
        peringatan.insert(0, "⚠️ PERINGATAN: Deskripsi iklan terdeteksi menyebut riwayat negatif (banjir/tabrak/kerusakan).")

    rekomendasi = {}
    if hasil['status'] == "Lambat":
        selisih_turun = max(0, hasil['harga_iklan'] - batas_lambat)
        selisih_cepat = max(0, hasil['harga_iklan'] - batas_cepat)
        rekomendasi = {
            "pesan": f"Turunkan minimal Rp {int(selisih_turun):,} ke bawah Rp {int(batas_lambat):,} agar masuk kategori Sedang, atau turunkan Rp {int(selisih_cepat):,} agar Cepat Laku.",
            "target_sedang": batas_lambat,
            "target_cepat": batas_cepat
        }
    elif hasil['status'] == "Sedang":
        selisih_cepat = max(0, hasil['harga_iklan'] - batas_cepat)
        rekomendasi = {
            "pesan": f"Harga Anda kompetitif di pasar. Turunkan sekitar Rp {int(selisih_cepat):,} (ke ≤ Rp {int(batas_cepat):,}) untuk mempercepat penjualan (kategori Cepat Laku).",
            "target_sedang": hasil['harga_iklan'],
            "target_cepat": batas_cepat
        }
    else:  # Cepat
        rekomendasi = {
            "pesan": "Harga sangat menarik dan berada di area kompetitif pasar. Berpotensi laku dalam waktu singkat!",
            "target_sedang": batas_lambat,
            "target_cepat": hasil['harga_iklan']
        }

    return {
        "status": hasil['status'],
        "proba": hasil['proba'],
        "harga_iklan": hasil['harga_iklan'],
        "harga_wajar": hasil['harga_wajar'],
        "sumber_harga_wajar": hasil['sumber_harga_wajar'],
        "deviasi_persen": round(hasil['deviasi_persen'], 2),
        "km_per_tahun": hasil['km_per_tahun'],
        "usia_mobil": hasil['usia_mobil'],
        "ada_garansi": hasil['ada_garansi'],
        "riwayat_negatif": hasil['riwayat_negatif'],
        "catatan_servis": hasil['catatan_servis'],
        "pajak_hidup": hasil['pajak_hidup'],
        "indikasi_kredit_tdp": hasil['indikasi_kredit_tdp'],
        "jumlah_sampel": hasil['jumlah_sampel'],
        "keyakinan_rendah": hasil['keyakinan_rendah'],
        "peringatan": peringatan,
        "rentang_pasar": {
            "q33_persen": Q33_TERTILE,
            "q67_persen": Q67_TERTILE,
            "batas_cepat_rp": batas_cepat,
            "batas_lambat_rp": batas_lambat
        },
        "rekomendasi": rekomendasi,
        "nlp_scan": nlp_scan
    }


@app.post("/api/simulate")
async def api_simulate(req: SimulateRequest):
    """Menghitung dampak penyesuaian harga terhadap kategori kecepatan laku lewat mesin_prediksi.py."""
    try:
        hasil = prediksi(
            sistem=sistem,
            merek=req.merek,
            model=req.model,
            tahun=req.tahun,
            transmisi=req.transmisi,
            km=req.km,
            harga_iklan=req.harga_simulasi,
            deskripsi=req.deskripsi or ""
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    selisih_dari_awal = req.harga_simulasi - req.harga_awal
    pct_perubahan = (selisih_dari_awal / req.harga_awal) * 100.0 if req.harga_awal else 0

    return {
        "status_baru": hasil['status'],
        "deviasi_baru": round(hasil['deviasi_persen'], 2),
        "harga_simulasi": req.harga_simulasi,
        "selisih_rp": selisih_dari_awal,
        "perubahan_persen": round(pct_perubahan, 2),
        "proba": hasil['proba']
    }


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 70)
    print("   🚗 AUTOLAKU FRONTEND SERVER (FastAPI + Jinja2 + Tailwind CSS)   ")
    print("   Buka di peramban: http://localhost:8000                        ")
    print("=" * 70 + "\n")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
