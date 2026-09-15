__all__ = [
    "create_figures",
    "create_public_harvest_figure",
    "create_puskesmas_figures",
    "create_surveillance_figures",
]


def __getattr__(name: str) -> object:
    if name == "create_public_harvest_figure":
        from dengue_st_diagnostics.artifacts.public_figures import (
            create_public_harvest_figure,
        )

        return create_public_harvest_figure
    if name in {"create_figures", "create_puskesmas_figures", "create_surveillance_figures"}:
        from dengue_st_diagnostics.artifacts import figures

        return getattr(figures, name)
    raise AttributeError(name)
