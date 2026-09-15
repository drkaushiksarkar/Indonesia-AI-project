from __future__ import annotations

from importlib import import_module

EXPORTS = {
    "harvest_boundaries": "boundaries",
    "harvest_ckan": "ckan",
    "harvest_gbif": "gbif",
    "harvest_kemkes": "kemkes",
    "harvest_literature": "literature",
    "harvest_ncbi": "ncbi",
    "harvest_opendengue": "opendengue",
    "harvest_portals": "portals",
    "harvest_public_sources": "public_sources",
    "harvest_puskesmas": "puskesmas",
    "harvest_satusehat": "satusehat",
    "harvest_who": "who",
}

__all__ = list(EXPORTS)


def __getattr__(name: str) -> object:
    module_name = EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(f"dengue_st_diagnostics.harvest.{module_name}")
    return getattr(module, name)
