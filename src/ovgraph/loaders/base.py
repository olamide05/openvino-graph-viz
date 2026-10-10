from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..graph import Graph


class Loader(ABC):
    """Turns a model file of one format into our framework-independent Graph."""
    name: str = "loader"
    extensions: tuple[str, ...] = ()

    @abstractmethod
    def load(self, path: Path) -> Graph: ...


_LOADERS: dict[str, Loader] = {}


def register_loader(cls: type[Loader]) -> type[Loader]:
    _LOADERS[cls.name] = cls()
    return cls


def available_loaders() -> list[str]:
    return sorted(_LOADERS)


def load_graph(path: str | Path, fmt: str | None = None) -> Graph:
    """Pick a loader by explicit name, else by file extension."""
    path = Path(path)
    if fmt is not None:
        if fmt not in _LOADERS:
            raise KeyError(f"unknown format {fmt!r}; available: {available_loaders()}")
        return _LOADERS[fmt].load(path)
    for loader in _LOADERS.values():
        if path.suffix.lower() in loader.extensions:
            return loader.load(path)
    raise ValueError(f"no loader for {path.suffix!r}; available: {available_loaders()}")
