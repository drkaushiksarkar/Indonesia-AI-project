"""Second pass: additional public provincial and regency CKAN catalogs."""

from concurrent.futures import ThreadPoolExecutor

from mine_catalogs import mine

CATALOGS = [
    "https://satudata.mempawahkab.go.id",
    "https://data.tanjabbarkab.go.id",
    "https://data.bojonegorokab.go.id",
    "https://data.bantulkab.go.id",
    "https://data.bandaacehkota.go.id",
    "https://data.bandung.go.id",
    "https://data.sumbarprov.go.id",
    "https://data.kalselprov.go.id",
    "https://data.kaltimprov.go.id",
    "https://data.sulselprov.go.id",
    "https://satudata.ntbprov.go.id",
    "https://data.nttprov.go.id",
    "https://data.baliprov.go.id",
    "https://data.jakarta.go.id",
    "https://data.jabarprov.go.id",
    "https://data.bantenprov.go.id",
    "https://data.kedirikota.go.id",
    "https://data.surakarta.go.id",
    "https://data.salatiga.go.id",
    "https://data.magelangkota.go.id",
    "https://data.purworejokab.go.id",
    "https://data.purbalinggakab.go.id",
    "https://data.cilacapkab.go.id",
    "https://data.banyumaskab.go.id",
    "https://satudata.kulonprogokab.go.id",
    "https://data.slemankab.go.id",
    "https://opendata.banyuwangikab.go.id",
    "https://data.surabaya.go.id",
]
if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(mine, CATALOGS))
