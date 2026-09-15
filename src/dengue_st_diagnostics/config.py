from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Paths:
    root: Path
    raw: Path
    interim: Path
    quantitative: Path
    images: Path

    @classmethod
    def create(cls, root: Path) -> Paths:
        paths = cls(
            root=root,
            raw=root / "raw",
            interim=root / "interim",
            quantitative=root / "quantitative",
            images=root / "images",
        )
        for value in paths.__dict__.values():
            value.mkdir(parents=True, exist_ok=True)
        return paths


@dataclass(frozen=True)
class Settings:
    values: dict[str, Any]
    paths: Paths

    @classmethod
    def load(cls, config_path: Path, output_root: Path | None = None) -> Settings:
        values = json.loads(config_path.read_text(encoding="utf-8"))
        configured = Path(values["output"]["root"])
        root = output_root if output_root is not None else configured
        return cls(values=values, paths=Paths.create(root.resolve()))

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.values[name])
