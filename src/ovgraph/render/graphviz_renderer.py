"""Graph -> Graphviz DOT -> SVG.

This is the only module that knows about Graphviz. It reads the graph and the
`meta` hints that passes left on nodes, and nothing else.
"""
from __future__ import annotations

import zlib
from html import escape
from pathlib import Path

import graphviz

from ..graph import Edge, Graph, Node, format_shape

DEFAULT_FILL = "#f2f2f2"
_PALETTE = ["#cfe8fc", "#d5f5d5", "#fde6c8", "#e9d9f7", "#fbd5d5",
            "#d8f0ee", "#f7f0c0", "#e0e0f0", "#f0dcc8", "#d9ead3"]
# Fixed colors for the most common ops; everything else gets a stable hash color.
FILL_BY_TYPE = {
    "Parameter": "#b7e1cd", "Result": "#f4c7c3", "Constant": "#d9d9d9",
    "MatMul": "#cfe8fc", "Add": "#fde6c8", "Convert": "#e9d9f7",
}


def _fill(op_type: str) -> str:
    if op_type in FILL_BY_TYPE:
        return FILL_BY_TYPE[op_type]
    return _PALETTE[zlib.crc32(op_type.encode()) % len(_PALETTE)]


def _cells(prefix: str, ports: list[int], total: int) -> str:
    """One <TD PORT=...> per port; the last cell stretches so the row spans `total` columns."""
    out = []
    for i, p in enumerate(ports):
        span = total - len(ports) + 1 if i == len(ports) - 1 else 1
        out.append(f'<TD PORT="{prefix}{p}" COLSPAN="{span}" BORDER="1">{p}</TD>')
    return "<TR>" + "".join(out) + "</TR>"


def node_label(node: Node, in_ports: list[int], out_ports: list[int],
               show_attributes: bool = False) -> str:
    total = max(len(in_ports), len(out_ports), 1)
    rows = []
    if in_ports:
        rows.append(_cells("in", in_ports, total))
    title = f"<B>{escape(node.name)}</B><BR/>{escape(node.op_type)} (id {node.id})"
    rows.append(f'<TR><TD COLSPAN="{total}">{title}</TD></TR>')
    for k, v in (node.attributes.items() if show_attributes else ()):
        rows.append(f'<TR><TD COLSPAN="{total}" ALIGN="LEFT">{escape(k)}={escape(v)}</TD></TR>')
    for _port, text in node.meta.get("extra_rows", []):
        rows.append(f'<TR><TD COLSPAN="{total}" ALIGN="LEFT">{escape(text)}</TD></TR>')
    if out_ports:
        rows.append(_cells("out", out_ports, total))
    return '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="3">' + "".join(rows) + "</TABLE>>"


def build_digraph(graph: Graph, edge_labels: bool = True,
                  show_attributes: bool = False) -> graphviz.Digraph:
    dot = graphviz.Digraph(format="svg", engine="dot")
    dot.attr(rankdir="TB", ranksep="0.3", nodesep="0.25")
    dot.attr("node", shape="box", style="filled", fontname="Tahoma", fontsize="10", margin="0")
    dot.attr("edge", fontname="Tahoma", fontsize="8")

    ins: dict[int, set[int]] = {i: set() for i in graph.nodes}
    outs: dict[int, set[int]] = {i: set() for i in graph.nodes}
    for e in graph.edges:
        outs[e.src_id].add(e.src_port)
        ins[e.dst_id].add(e.dst_port)

    for n in graph.nodes.values():
        color = n.meta.get("border_color", "black")
        pen = "3" if "border_color" in n.meta else "1"
        dot.node(str(n.id), label=node_label(n, sorted(ins[n.id]), sorted(outs[n.id]), show_attributes),
                 fillcolor=_fill(n.op_type), color=color, penwidth=pen)

    for e in graph.edges:
        kw = {}
        if edge_labels:
            kw["label"] = f"{e.dtype or '?'}{format_shape(e.shape)}"
        dot.edge(f"{e.src_id}:out{e.src_port}:s", f"{e.dst_id}:in{e.dst_port}:n", **kw)
    return dot


class GraphvizRenderer:
    def __init__(self, edge_labels: bool = True, show_attributes: bool = False) -> None:
        self.edge_labels = edge_labels
        self.show_attributes = show_attributes

    def to_dot(self, graph: Graph) -> str:
        return build_digraph(graph, self.edge_labels, self.show_attributes).source

    def render_svg(self, graph: Graph, path: str | Path | None = None) -> bytes:
        svg = build_digraph(graph, self.edge_labels, self.show_attributes).pipe(format="svg")
        if path is not None:
            Path(path).write_bytes(svg)
        return svg
