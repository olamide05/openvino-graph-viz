from __future__ import annotations

from typing import Callable

from ..graph import Graph, Node
from .base import Pass, register_pass


@register_pass
class HighlightPass(Pass):
    """Give selected nodes a red border.

    Select by node NAMES (the assignment's main way), by node ids, by op types, or
    with any predicate. If several are given a node is highlighted when ANY of them
    matches. Names are not unique, so every node with a listed name is highlighted.
    """
    name = "highlight"

    def __init__(self, names: set[str] | list[str] | None = None,
                 node_ids: set[int] | None = None,
                 op_types: set[str] | None = None,
                 predicate: Callable[[Node], bool] | None = None,
                 color: str = "red") -> None:
        self.names = set(names or ())
        self.node_ids = set(node_ids or ())
        self.op_types = set(op_types or ())
        self.predicate = predicate
        self.color = color

    def _match(self, n: Node) -> bool:
        return (n.name in self.names
                or n.id in self.node_ids
                or n.op_type in self.op_types
                or (self.predicate is not None and self.predicate(n)))

    @classmethod
    def from_options(cls, options: dict[str, list[str]]) -> "HighlightPass":
        """Options: names=a,b  ids=1,2  ops=MatMul,Add  color=blue (every key optional)."""
        unknown = set(options) - {"names", "ids", "ops", "color"}
        if unknown:
            raise ValueError(f"highlight: unknown option(s) {', '.join(sorted(unknown))}; "
                             "use names, ids, ops, color")
        try:
            ids = {int(i) for i in options.get("ids", [])}
        except ValueError:
            raise ValueError("highlight: ids must be integers") from None
        color = options.get("color", ["red"])[-1]
        return cls(names=set(options.get("names", [])), node_ids=ids,
                   op_types=set(options.get("ops", [])), color=color)

    def run(self, graph: Graph) -> Graph:
        for n in graph.nodes.values():
            if self._match(n):
                n.meta["border_color"] = self.color
        return graph
