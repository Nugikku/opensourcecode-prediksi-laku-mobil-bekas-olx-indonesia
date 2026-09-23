import asyncio
from playwright.async_api import async_playwright
import pandas as pd
import re

async def scrape_olx(target_data=3000):
    all_cars = []
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        # Menetapkan ukuran layar standar agar tombol tidak tersembunyi
        context = await browser.new_context(
            viewport={'width': 1366, 'height': 768},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        # --- LOGIKA: Tangkap Data Tanpa Memperdulikan Nama URL-nya ---
        async def tangkap_response(response):
            # Hanya periksa respons yang merupakan paket data JSON
            if "application/json" in response.headers.get("content-type", ""):
                try:
                    res_json = await response.json()
                    # Pastikan paket data ini berisi daftar/list informasi (bukan data analitik)
                    if isinstance(res_json, dict) and "data" in res_json and isinstance(res_json["data"], list):
                        items = res_json.get("data", [])
                        for item in items:
                            # Filter ketat: Pastikan iklan khusus kategori Mobil Bekas (category_id 198)
                            cat_id = str(item.get("category_id", ""))
                            param_dict = {}
                            for param in item.get("parameters", []):
                                param_dict[param.get("key")] = param.get("value_name", param.get("value"))

                            # Ciri-ciri iklan mobil asli: Memiliki category_id '198' atau memiliki parameter 'make' & 'year'
                            is_car = (cat_id == "198") or ("make" in param_dict and "year" in param_dict)
                            
                            # Filter tambahan: Eliminasi merek non-mobil (seperti Apple, Samsung, dll.)
                            merek_non_mobil = ["apple", "samsung", "xiaomi", "oppo", "vivo", "realme", "asus", "lenovo"]
                            if param_dict.get("make", "").lower() in merek_non_mobil:
                                is_car = False

                            if "parameters" in item and "price" in item and is_car:
                                car = {
                                    "id_iklan": item.get("id"),
                                    "created_at": item.get("created_at"),  # Tanggal & waktu iklan dibuat
                                    "judul": item.get("title"),
                                    "deskripsi": item.get("description"),
                                    "merek": param_dict.get("make", "Lainnya"),
                                    "model": param_dict.get("model", "Lainnya"),
                                    "tahun": param_dict.get("year"),
                                    "transmisi": param_dict.get("transmission", "Lainnya"),
                                    "jarak_tempuh": param_dict.get("mileage"),
                                    "harga": item.get("price", {}).get("value", {}).get("raw"),
                                    "tipe_penjual_badge": item.get("user", {}).get("badge", "Individu"),
                                    "lokasi": item.get("locations_resolved", {}).get("ADMIN_LEVEL_1_name", "Lainnya")
                                }
                                
                                if car["id_iklan"] and car["id_iklan"] not in [c["id_iklan"] for c in all_cars]:
                                    all_cars.append(car)
                except Exception:
                    pass

        # Daftar URL lengkap dengan pembagian per merek dan wilayah
        urls = [
            # Kategori Utama
            "https://www.olx.co.id/mobil-bekas_c198",
            # Kategori Berdasarkan Merek Populer
            "https://www.olx.co.id/mobil-bekas_c198/toyota_m53",
            "https://www.olx.co.id/mobil-bekas_c198/honda_m27",
            "https://www.olx.co.id/mobil-bekas_c198/daihatsu_m19",
            "https://www.olx.co.id/mobil-bekas_c198/suzuki_m52",
            "https://www.olx.co.id/mobil-bekas_c198/mitsubishi_m42",
            # Kategori Berdasarkan Wilayah
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

        for idx, target_url in enumerate(urls, 1):
            if len(all_cars) >= target_data:
                print(f"Target {target_data} data tercapai!")
                break

            print(f"\n[{idx}/{len(urls)}] Membuka URL: {target_url}")
            try:
                await page.goto(target_url, wait_until="domcontentloaded")
                await asyncio.sleep(4)
            except Exception as err:
                print(f"Gagal membuka URL: {err}")
                continue

            percobaan_kosong = 0
            while len(all_cars) < target_data and percobaan_kosong < 8:
                jumlah_sebelum = len(all_cars)
                
                # 1. Scroll bertahap ke bawah
                for _ in range(4):
                    await page.evaluate("window.scrollBy(0, 1000);")
                    await asyncio.sleep(1.0)
                
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
                await asyncio.sleep(1.5)

                # 2. Cari & klik tombol "Muat lainnya"
                try:
                    tombol = page.locator('button[data-aut-id="btnLoadMore"], button:has-text("Muat lainnya"), button:has-text("Load more")').first
                    if await tombol.count() > 0 and await tombol.is_visible():
                        await tombol.scroll_into_view_if_needed()
                        await tombol.click(force=True)
                        print("--> Klik 'Muat lainnya'...")
                        await asyncio.sleep(3.0)
                except Exception:
                    pass

                print(f"Total Terkumpul: {len(all_cars)} / {target_data} data mobil...")

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
    print("Memulai scraping OLX via Browser...")
    hasil = asyncio.run(scrape_olx(target_data=3000))
    
    if not hasil:
        print("\nData masih kosong. Sistem keamanan OLX mungkin sedang sangat ketat di IP jaringan Anda.")
    else:
        df = pd.DataFrame(hasil)
        df = df.drop_duplicates(subset=["id_iklan"]).reset_index(drop=True)
        
        df.to_csv("dataset_olx_mentah.csv", index=False, encoding="utf-8-sig")
        print(f"\nSelesai! Berhasil menyimpan {len(df)} baris ke 'dataset_olx_mentah.csv'")
        print(df[["created_at", "judul", "merek", "tahun", "harga", "tipe_penjual_badge", "lokasi"]].head())