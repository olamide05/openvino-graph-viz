from __future__ import annotations

from abc import ABC, abstractmethod

from ..graph import Graph


class Pass(ABC):
    """A graph transformation. Subclass it, implement `run`, done.

    `run` receives a graph it OWNS (the pipeline cloned it), so a pass may
    mutate it freely and return it.
    """
    name: str = "pass"

    @abstractmethod
    def run(self, graph: Graph) -> Graph: ...


class Pipeline:
    """Runs passes in order, always on a clone of the input graph."""

    def __init__(self, passes: list[Pass] | None = None) -> None:
        self.passes: list[Pass] = list(passes or [])

    def add(self, p: Pass) -> "Pipeline":
        self.passes.append(p)
        return self

    def run(self, graph: Graph) -> Graph:
        g = graph.clone()
        for p in self.passes:
            g = p.run(g)
        return g


# Registry: lets the CLI pick passes by name, and lets third-party code add
# passes without editing this package (just decorate the class).
_REGISTRY: dict[str, type[Pass]] = {}


def register_pass(cls: type[Pass]) -> type[Pass]:
    _REGISTRY[cls.name] = cls
    return cls


def get_pass(name: str) -> type[Pass]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown pass {name!r}; available: {sorted(_REGISTRY)}") from None


def available_passes() -> list[str]:
    return sorted(_REGISTRY)
