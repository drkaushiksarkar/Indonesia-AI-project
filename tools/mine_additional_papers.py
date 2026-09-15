import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mine_paper_links import fetch

URLS = [
    "https://ejurnalmalahayati.ac.id/index.php/kesehatan/article/download/18958/pdf",
    "https://journal.ugm.ac.id/bik/article/download/55908/31370",
    "https://repository.poltekkes-smg.ac.id/repository/ADI%20SEPTIAN%2016.pdf",
    "https://assets-eu.researchsquare.com/files/rs-7043263/v1/7aa0785a-8a59-4432-9e69-9ea0254a80dc.pdf?c=1754906966",
    "https://microbiol.crie.ru/jour/article/view/18631",
    "https://repository.its.ac.id/101197/",
    "https://www.mdpi.com/2227-9717/10/11/2454",
    "https://repository.ung.ac.id/get/simlit/2/949/1/KAJIAN-FAKTOR-LINGKUNGAN-TERHADAP-KASUS-DEMAM-BERDARAH-DENGUE-DBD-STUDI-KASUS-DI-KOTA-GORONTALO-PROVINSI-GORONTALO.pdf",
]
if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, URLS))
    Path("../outputs/extensive_mining/literature/additional_manifest.json").write_text(
        json.dumps(results, indent=2)
    )
    for r in results:
        print(r["url"], r["status"], r.get("links", []), flush=True)
