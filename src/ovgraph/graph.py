"""Framework-independent graph model.

Nothing in this file knows about OpenVINO or Graphviz. Loaders produce a
`Graph`, passes transform it, renderers draw it.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

# One dimension is an int, or None when it is dynamic (printed as "?").
Shape = tuple["int | None", ...]


def format_shape(shape: Shape | None) -> str:
    """[1,?,768]. Missing shape metadata prints as '?' rather than crashing."""
    if shape is None:
        return "?"
    return "[" + ",".join("?" if d is None else str(d) for d in shape) + "]"


@dataclass
class Node:
    # Unique ID, separate from `name` because model names can repeat.
    id: int
    name: str
    op_type: str
    attributes: dict[str, str] = field(default_factory=dict)
    # Visualization metadata that passes write and the renderer reads,
    # e.g. {"border_color": "red", "extra_rows": [(port, text)]}.
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    """Data flowing from one output port to one input port.

    Two nodes can be joined by several edges (different ports), so edges
    are kept in a list, never in a dict keyed by (src, dst).
    """
    src_id: int
    src_port: int
    dst_id: int
    dst_port: int
    shape: Shape | None = None
    dtype: str | None = None
    # Text to draw on the edge instead of "dtype[shape]". Passes that merge
    # several edges into one use it (e.g. "12 edges").
    label: str | None = None


class Graph:
    def __init__(self) -> None:
        self.nodes: dict[int, Node] = {}
        self.edges: list[Edge] = []

    def add_node(self, node: Node) -> Node:
        if node.id in self.nodes:
            raise ValueError(f"duplicate node id {node.id}")
        self.nodes[node.id] = node
        return node

    def add_edge(self, edge: Edge) -> Edge:
        for nid in (edge.src_id, edge.dst_id):
            if nid not in self.nodes:
                raise KeyError(f"edge refers to unknown node id {nid}")
        self.edges.append(edge)
        return edge

    def out_edges(self, node_id: int) -> list[Edge]:
        return [e for e in self.edges if e.src_id == node_id]

    def in_edges(self, node_id: int) -> list[Edge]:
        return [e for e in self.edges if e.dst_id == node_id]

    def clone(self) -> Graph:
        """Deep copy. Passes work on a clone so the loaded model stays intact."""
        g = Graph()
        g.nodes = copy.deepcopy(self.nodes)
        g.edges = list(self.edges)  # Edge is frozen, so sharing is safe
        return g

    def __len__(self) -> int:
        return len(self.nodes)
