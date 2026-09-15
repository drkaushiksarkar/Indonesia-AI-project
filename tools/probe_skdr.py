"""Read public dashboard chart using the same anonymous session/form as its UI."""

from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path("../outputs/extensive_mining/dashboards")
if __name__ == "__main__":
    s = requests.Session()
    base = "https://surkarkes.kemkes.go.id"
    r = s.get(base + "/dashboard-skdr-dev", timeout=45)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    token = soup.select_one('meta[name="csrf-token"]')["content"]
    data = {
        "_csrf": token,
        "kode_penyakit": "16SDF",
        "tahun": "2023",
        "kd_prop": "33",
        "kd_kab": "",
        "jenis_waktu": "1",
        "minggu1": "",
        "minggu2": "",
        "tahun1": "",
        "tahun2": "",
        "multi_penyakit": "0",
    }
    r = s.post(base + "/index.php/api-skk/grafik-suspek-ibs", data=data, timeout=60)
    print(r.status_code, len(r.content))
    (ROOT / "skdr_probe.html").write_text(r.text)
    r = s.post(base + "/index.php/kabupaten-list", data={"_csrf": token, "prov": "33"}, timeout=40)
    print("districts", r.status_code, r.text[:2000])
    (ROOT / "skdr_districts33.html").write_text(r.text)
