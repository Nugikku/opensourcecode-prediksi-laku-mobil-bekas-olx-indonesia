"""
Kamus normalisasi slang, singkatan istilah jual-beli, dan kosakata otomotif OLX.
Digunakan untuk standardisasi teks sebelum proses ekstraksi fitur (NLP) dan pemodelan.
"""

KAMUS_SLANG_OTOMOTIF = {
    # --- 1. Transmisi, Bahan Bakar & Mesin ---
    "mt": "manual", "m/t": "manual", "mnl": "manual", "manul": "manual",
    "at": "otomatis", "a/t": "otomatis", "matic": "otomatis", "matict": "otomatis", 
    "metik": "otomatis", "metuk": "otomatis", "triptonic": "otomatis", "tiptronic": "otomatis",
    "cc": "kapasitas mesin", "silinder": "kapasitas mesin",
    "bsn": "bensin", "bensin": "bensin", "dsl": "diesel", "solar": "diesel",
    "enginer": "mesin", "engine": "mesin", "msn": "mesin", "kering": "mesin kering",

    # --- 2. Surat-Surat, Pajak, Kepemilikan & Legalitas ---
    "stnk": "stnk", "bpkb": "bpkb", "faktur": "faktur", "nik": "tahun perakitan",
    "pjk": "pajak", "pajak": "pajak", "pjkny": "pajak", "pjg": "panjang", 
    "pnjg": "panjang", "puanjang": "panjang", "on": "hidup", "off": "mati",
    "komplit": "lengkap", "kmplit": "lengkap", "lkp": "lengkap", "komplit2": "lengkap",
    "surat2": "surat lengkap", "keabsahan": "legalitas terjamin",
    "tgn": "tangan", "tgn1": "tangan pertama", "tgn2": "tangan kedua", "tg1": "tangan pertama",
    "an": "atas nama", "a/n": "atas nama", "perorangan": "individu", "anperorangan": "milik pribadi",
    "plat": "plat nomor", "kaleng": "masa berlaku plat", "ganjil": "plat ganjil", "genap": "plat genap",
    "pmk": "tahun pemakaian", "pemakaian": "tahun pemakaian",

    # --- 3. Riwayat Perawatan, Odometer & Kondisi Fisik ---
    "km": "kilometer", "odo": "odometer", "low": "rendah", "lowkm": "kilometer rendah",
    "rec": "catatan servis", "record": "catatan servis", "bengkelresmi": "bengkel resmi",
    "ori": "orisinil", "orisinil": "orisinil", "orisinal": "orisinil", "ors": "orisinil", "original": "orisinil",
    "cat": "cat bodi", "mulus": "mulus", "mlus": "mulus", "kinclong": "bersih", "gress": "seperti baru", "mint": "seperti baru",
    "ac": "pendingin kabin", "dngn": "dingin", "nyess": "dingin", "menggigil": "sangat dingin",
    "kaki": "suspensi", "kaki2": "suspensi", "senyap": "hening", "anteng": "stabil",
    "ban": "ban", "tebel": "tebal", "tbl": "tebal",
    "siap": "siap", "siap2": "siap", "pke": "pakai", "pakai": "pakai", "gas": "jalan",
    "istmw": "istimewa", "istmwa": "istimewa", "antik": "sangat terawat", "terawat": "sangat terawat",

    # --- 4. Integritas Unit, Bebas Insiden & Sertifikasi ---
    "nobanjir": "bebas banjir", "nolaka": "bebas tabrak", "bebasbanjir": "bebas banjir", "bebastabrak": "bebas tabrak",
    "grnsi": "garansi", "garansi": "garansi", "warranty": "garansi",
    "sertif": "sertifikat", "sertifikasi": "sertifikat", "inspeksi": "inspeksi", "terinspeksi": "lulus inspeksi",
    "otospector": "inspeksi otospector", "olxmobbi": "inspeksi bersertifikat",

    # --- 5. Finansial, Skema Transaksi & Dealer/Showroom ---
    "nego": "negosiasi", "nggo": "negosiasi", "nett": "harga pas", "net": "harga pas",
    "halus": "negosiasi tipis", "tipis": "negosiasi tipis", "sadis": "penawaran rendah",
    "dijual": "jual", "djual": "jual", "bu": "butuh uang", "b.u": "butuh uang",
    "tt": "tukar tambah", "bt": "barter", "kredit": "kredit", "krdt": "kredit",
    "cash": "tunai", "csh": "tunai", "dp": "uang muka", "tdp": "total uang muka",
    "angs": "angsuran", "angsrn": "angsuran", "cicilan": "angsuran",
    "otr": "harga jalan", "srw": "showroom", "sr": "showroom", "leasing": "perusahaan pembiayaan",

    # --- 6. Kata Sambung & Slang Percakapan Harian ---
    "yg": "yang", "dgn": "dengan", "dg": "dengan", "utk": "untuk", "untk": "untuk",
    "tp": "tapi", "krn": "karena", "karna": "karena", "jd": "jadi", "jgn": "jangan",
    "udh": "sudah", "udah": "sudah", "dah": "sudah", "sdh": "sudah",
    "blm": "belum", "trs": "terus", "sy": "saya", "kamu": "kamu",
    "bs": "bisa", "gk": "tidak", "ga": "tidak", "gak": "tidak", "tdk": "tidak",
    "bgt": "banget", "bener": "benar", "aja": "saja", "kalo": "kalau", "lok": "lokasi"
}