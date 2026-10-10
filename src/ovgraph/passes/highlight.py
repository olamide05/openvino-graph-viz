from __future__ import annotations

from typing import Callable

from ..graph import Graph, Node
from .base import Pass, register_pass


@register_pass
class HighlightPass(Pass):
    """Give selected nodes a red border.

    Select by node ids, by op types, or with any predicate. If several are
    given a node is highlighted when ANY of them matches.
    """
    name = "highlight"

    def __init__(self, node_ids: set[int] | None = None,
                 op_types: set[str] | None = None,
                 predicate: Callable[[Node], bool] | None = None,
                 color: str = "red") -> None:
        self.node_ids = set(node_ids or ())
        self.op_types = set(op_types or ())
        self.predicate = predicate
        self.color = color

    def _match(self, n: Node) -> bool:
        return (n.id in self.node_ids
                or n.op_type in self.op_types
                or (self.predicate is not None and self.predicate(n)))

    def run(self, graph: Graph) -> Graph:
        for n in graph.nodes.values():
            if self._match(n):
                n.meta["border_color"] = self.color
        return graph
