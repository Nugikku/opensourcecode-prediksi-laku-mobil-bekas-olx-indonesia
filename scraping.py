import asyncio
from playwright.async_api import async_playwright
import pandas as pd
import re
import os

TARGET_TOTAL_DATA = 1000
OUTPUT_FILE_RAW = "dataset_olx_mentah.csv"

def deteksi_indikasi_terjual(judul, deskripsi):
    """Mendeteksi apakah iklan sudah ditandai terjual oleh penjual."""
    teks = f"{str(judul)} {str(deskripsi)[:200]}".lower()
    kata_terjual = [
        r'\bterjual\b', r'\bsold\b', r'\blaku\b', 
        r'\bbooked\b', r'\bdp masuk\b', r'\bsudah laku\b'
    ]
    for pola in kata_terjual:
        if re.search(pola, teks):
            return "Terjual"
    return "Tersedia"

async def scrape_olx(target_data=TARGET_TOTAL_DATA):
    all_cars = []
    id_terekam = set()

    # Muat data yang sudah tersimpan jika script dijalankan ulang
    if os.path.exists(OUTPUT_FILE_RAW):
        try:
            df_ada = pd.read_csv(OUTPUT_FILE_RAW)
            if "id_iklan" in df_ada.columns:
                all_cars = df_ada.to_dict("records")
                id_terekam = set(df_ada["id_iklan"].dropna().astype(str).tolist())
                print(f"[INFO] Melanjutkan progres: {len(all_cars)} data sudah ada di {OUTPUT_FILE_RAW}")
        except Exception:
            pass

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            viewport={'width': 1366, 'height': 768},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        async def tangkap_response(response):
            if "application/json" in response.headers.get("content-type", ""):
                try:
                    res_json = await response.json()
                    if isinstance(res_json, dict) and "data" in res_json and isinstance(res_json["data"], list):
                        items = res_json.get("data", [])
                        for item in items:
                            cat_id = str(item.get("category_id", ""))
                            param_dict = {}
                            for param in item.get("parameters", []):
                                param_dict[param.get("key")] = param.get("value_name", param.get("value"))

                            is_car = (cat_id == "198") or ("make" in param_dict and "year" in param_dict)
                            
                            merek_non_mobil = ["apple", "samsung", "xiaomi", "oppo", "vivo", "realme", "asus", "lenovo"]
                            if str(param_dict.get("make", "")).lower() in merek_non_mobil:
                                is_car = False

                            iklan_id = str(item.get("id", ""))
                            if "parameters" in item and "price" in item and is_car and iklan_id:
                                if iklan_id not in id_terekam:
                                    judul_raw = item.get("title", "")
                                    deskripsi_raw = item.get("description", "")
                                    
                                    car = {
                                        "id_iklan": iklan_id,
                                        "tanggal_posting": item.get("created_at"),
                                        "status_iklan_terjual": deteksi_indikasi_terjual(judul_raw, deskripsi_raw),
                                        "judul": judul_raw,
                                        "deskripsi": deskripsi_raw,
                                        "merek": param_dict.get("make", "Lainnya"),
                                        "model": param_dict.get("model", "Lainnya"),
                                        "tahun": param_dict.get("year"),
                                        "transmisi": param_dict.get("transmission", "Lainnya"),
                                        "jarak_tempuh": param_dict.get("mileage"),
                                        "harga": item.get("price", {}).get("value", {}).get("raw"),
                                        "tipe_penjual_badge": item.get("user", {}).get("badge", "Individu"),
                                        "lokasi": item.get("locations_resolved", {}).get("ADMIN_LEVEL_1_name", "Lainnya")
                                    }
                                    all_cars.append(car)
                                    id_terekam.add(iklan_id)
                except Exception:
                    pass

        urls = [
            "https://www.olx.co.id/mobil-bekas_c198",
            "https://www.olx.co.id/mobil-bekas_c198/toyota_m53",
            "https://www.olx.co.id/mobil-bekas_c198/honda_m27",
            "https://www.olx.co.id/mobil-bekas_c198/daihatsu_m19",
            "https://www.olx.co.id/mobil-bekas_c198/suzuki_m52",
            "https://www.olx.co.id/mobil-bekas_c198/mitsubishi_m42",
            "https://www.olx.co.id/jakarta-dki_g2000007/mobil-bekas_c198",
            "https://www.olx.co.id/jawa-barat_g2000008/mobil-bekas_c198",
            "https://www.olx.co.id/jawa-timur_g2000010/mobil-bekas_c198",
            "https://www.olx.co.id/jawa-tengah_g2000009/mobil-bekas_c198",
            "https://www.olx.co.id/banten_g2000003/mobil-bekas_c198",
            "https://www.olx.co.id/sumatera-utara_g2000022/mobil-bekas_c198",
            "https://www.olx.co.id/bali_g2000002/mobil-bekas_c198",
            "https://www.olx.co.id/yogyakarta-di_g2000033/mobil-bekas_c198",
            "https://www.olx.co.id/sulawesi-selatan_g2000025/mobil-bekas_c198",
            "https://www.olx.co.id/riau_g2000020/mobil-bekas_c198",
            "https://www.olx.co.id/sumatera-selatan_g2000021/mobil-bekas_c198",
            "https://www.olx.co.id/lampung_g2000015/mobil-bekas_c198",
            "https://www.olx.co.id/kalimantan-timur_g2000014/mobil-bekas_c198",
            "https://www.olx.co.id/kalimantan-barat_g2000011/mobil-bekas_c198",
            "https://www.olx.co.id/kalimantan-selatan_g2000012/mobil-bekas_c198",
            "https://www.olx.co.id/nusa-tenggara-barat_g2000017/mobil-bekas_c198"
        ]

        page.on("response", tangkap_response)
        checkpoint_simpan = len(all_cars)

        for idx, target_url in enumerate(urls, 1):
            if len(all_cars) >= target_data:
                print(f"\n[SUKSES] Target {target_data} data tercapai!")
                break

            print(f"\n[{idx}/{len(urls)}] Membuka URL: {target_url}")
            try:
                await page.goto(target_url, wait_until="domcontentloaded", timeout=45000)
                await asyncio.sleep(4)
            except Exception as err:
                print(f"[!] Gagal membuka URL ({err}), melanjutkan ke URL berikutnya...")
                continue

            percobaan_kosong = 0
            while len(all_cars) < target_data and percobaan_kosong < 8:
                jumlah_sebelum = len(all_cars)
                
                for _ in range(4):
                    await page.evaluate("window.scrollBy(0, 1000);")
                    await asyncio.sleep(1.0)
                
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
                await asyncio.sleep(1.5)

                try:
                    tombol = page.locator('button[data-aut-id="btnLoadMore"], button:has-text("Muat lainnya"), button:has-text("Load more")').first
                    if await tombol.count() > 0 and await tombol.is_visible():
                        await tombol.scroll_into_view_if_needed()
                        await tombol.click(force=True)
                        print("--> Klik 'Muat lainnya'...")
                        await asyncio.sleep(3.0)
                except Exception:
                    pass

                print(f"Total Terkumpul: {len(all_cars)} / {target_data} baris...")

                # Auto-save berkala setiap kelipatan 250 data
                if len(all_cars) - checkpoint_simpan >= 250:
                    pd.DataFrame(all_cars).to_csv(OUTPUT_FILE_RAW, index=False, encoding="utf-8-sig")
                    checkpoint_simpan = len(all_cars)
                    print(f"[Auto-Checkpoint] Tersimpan sementara: {checkpoint_simpan} baris.")

                if len(all_cars) == jumlah_sebelum:
                    percobaan_kosong += 1
                    if percobaan_kosong % 2 == 0:
                        await page.evaluate("window.scrollBy(0, -1200);")
                        await asyncio.sleep(1.0)
                        await page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
                        await asyncio.sleep(1.5)
                else:
                    percobaan_kosong = 0

        await browser.close()

    return all_cars[:target_data]

if __name__ == "__main__":
    print("Memulai scraping OLX dengan atribut tanggal posting & indikasi terjual...")
    hasil = asyncio.run(scrape_olx(target_data=TARGET_TOTAL_DATA))
    
    if not hasil:
        print("\nData kosong. Periksa koneksi internet atau status limit IP.")
    else:
        df = pd.DataFrame(hasil)
        df = df.drop_duplicates(subset=["id_iklan"]).reset_index(drop=True)
        df.to_csv(OUTPUT_FILE_RAW, index=False, encoding="utf-8-sig")
        print(f"\nSelesai! Berhasil menyimpan {len(df)} baris ke '{OUTPUT_FILE_RAW}'")
        kolom_cek = ["tanggal_posting", "status_iklan_terjual", "merek", "tahun", "harga", "tipe_penjual_badge", "lokasi"]
        print(df[kolom_cek].head())