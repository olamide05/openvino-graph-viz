from __future__ import annotations

from abc import ABC, abstractmethod

from ..graph import Graph


class Pass(ABC):
    """A graph transformation. Subclass it, implement `run`, done.

    `run` receives a graph it OWNS (the pipeline cloned it), so a pass may
    mutate it freely and return it.
    """
    name: str = "pass"
    # A pipeline skips passes with enabled = False (so a pass can be switched off without removing it).
    enabled: bool = True

    @abstractmethod
    def run(self, graph: Graph) -> Graph: ...

    @classmethod
    def from_options(cls, options: dict[str, list[str]]) -> "Pass":
        """Build the pass from CLI-style options ({key: [values]}). Override to accept options."""
        if options:
            raise ValueError(f"pass {cls.name!r} takes no options (got {', '.join(sorted(options))})")
        return cls()


class Pipeline:
    """Runs passes in order, always on a clone of the input graph."""

    def __init__(self, passes: list[Pass] | None = None) -> None:
        self.passes: list[Pass] = list(passes or [])

    def add(self, p: Pass) -> "Pipeline":
        self.passes.append(p)
        return self

    def set_enabled(self, name: str, enabled: bool) -> "Pipeline":
        """Enable or disable every pass called `name` without removing it."""
        for p in self.passes:
            if p.name == name:
                p.enabled = enabled
        return self

    def run(self, graph: Graph) -> Graph:
        g = graph.clone()
        for p in self.passes:
            if p.enabled:
                g = p.run(g)
        return g


# Registry: lets the CLI pick passes by name, and lets third-party code add
# passes without editing this package (just decorate the class).
_REGISTRY: dict[str, type[Pass]] = {}


def register_pass(cls: type[Pass]) -> type[Pass]:
    _REGISTRY[cls.name] = cls
    return cls


def get_pass(name: str) -> type[Pass]:
    """Look a pass up by name. 'hide-constants' and 'hide_constants' are the same."""
    try:
        return _REGISTRY[name.replace("-", "_")]
    except KeyError:
        raise KeyError(f"unknown pass {name!r}; available: {sorted(_REGISTRY)}") from None


def available_passes() -> list[str]:
    return sorted(_REGISTRY)
