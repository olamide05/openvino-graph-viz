"""Write the information the picture leaves out (names, attributes, collapsed edges) to a JSON file.

The picture can stay small because everything it hides is still here, looked up by node id.
"""
from __future__ import annotations

import json
from pathlib import Path

from .graph import Graph, format_shape


def _edge_info(e) -> dict:
    return {"src_id": e.src_id, "src_port": e.src_port, "dst_id": e.dst_id, "dst_port": e.dst_port,
            "dtype": e.dtype, "shape": format_shape(e.shape), "label": e.label}


def node_map(graph: Graph) -> dict[str, dict]:
    """{node id (as text) -> everything known about that node}. JSON keys must be text."""
    out: dict[str, dict] = {}
    for n in graph.nodes.values():
        entry: dict = {"name": n.name, "op_type": n.op_type, "attributes": n.attributes}
        rows = n.meta.get("extra_rows")
        if rows:
            entry["rows"] = [text for _port, text in rows]
        # Edges that collapse_fanout removed from the picture; the full connectivity lives here.
        collapsed = n.meta.get("collapsed_edges")
        if collapsed:
            entry["collapsed_outputs"] = [_edge_info(e) for e in collapsed]
        # Nodes that collapse_modules folded into this group box.
        members = n.meta.get("members")
        if members:
            entry["members"] = members
        # Edges inside a module box, and edges between a box and the rest, with the real node ids.
        folded = n.meta.get("folded_edges")
        if folded:
            entry["folded_edges"] = [_edge_info(e) for e in folded]
        out[str(n.id)] = entry
    return out


def write_node_map(graph: Graph, path: str | Path) -> None:
    Path(path).write_text(json.dumps(node_map(graph), indent=2), encoding="utf-8")
