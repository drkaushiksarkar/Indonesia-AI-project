__all__ = ["run_pipeline"]


def __getattr__(name: str) -> object:
    if name == "run_pipeline":
        from dengue_st_diagnostics.pipeline import run_pipeline

        return run_pipeline
    raise AttributeError(name)
