from __future__ import annotations

from ..graph import Graph, format_shape
from .base import Pass, register_pass

CONSTANT = "Constant"


@register_pass
class HideConstantsPass(Pass):
    """Remove Constant nodes and show them as rows on the node that uses them.

    A consumer fed by a constant on input port 1 gets the row
        [1]: f32[768,768]
    If the same constant feeds several consumers each row also carries
        (shared: <constant node id>)
    Only dtype and shape are shown, never tensor values.
    """
    name = "hide_constants"

    def __init__(self, constant_type: str = CONSTANT) -> None:
        self.constant_type = constant_type

    def run(self, graph: Graph) -> Graph:
        const_ids = {i for i, n in graph.nodes.items()
                     if n.op_type == self.constant_type}

        # One pass up front: how many distinct (consumer, port) pairs each constant feeds.
        # (Calling graph.out_edges() per edge here made the pass O(E^2) on shared constants.)
        users: dict[int, set[tuple[int, int]]] = {c: set() for c in const_ids}
        for e in graph.edges:
            if e.src_id in const_ids:
                users[e.src_id].add((e.dst_id, e.dst_port))

        kept = []
        for e in graph.edges:
            if e.src_id in const_ids:
                if e.dst_id in const_ids:
                    continue  # constant -> constant: both disappear
                n_users = len(users[e.src_id])
                text = f"[{e.dst_port}]: {e.dtype or '?'}{format_shape(e.shape)}"
                if n_users > 1:
                    text += f" (shared: {e.src_id})"
                graph.nodes[e.dst_id].meta.setdefault("extra_rows", []).append(
                    (e.dst_port, text))
            elif e.dst_id not in const_ids:
                kept.append(e)

        for cid in const_ids:
            del graph.nodes[cid]
        graph.edges = kept

        for n in graph.nodes.values():  # keep rows in port order
            if "extra_rows" in n.meta:
                n.meta["extra_rows"].sort(key=lambda r: r[0])
        return graph
