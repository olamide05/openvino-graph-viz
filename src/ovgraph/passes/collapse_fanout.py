"""Pass that stops drawing the edges of nodes with many consumers.
"""
from __future__ import annotations

from collections import Counter

from ..graph import Graph, format_shape
from .base import Pass, register_pass


@register_pass
class CollapseFanoutPass(Pass):
    """Replace the edges of a node with more than `threshold` outgoing edges by small text rows.

    In TinyLlama a few shared inputs (the attention mask, the position ids, beam_idx)
    feed all 22 layers. Their edges are a few percent of the edge count but most of the
    drawn edge LENGTH, so they cross the whole picture.

    Connectivity is not lost, it moves into the rows (same idea as hide_constants):
      on the source:    out[0] -> 44 consumers (collapsed)
      on each consumer: [1]: from id 9: i64[1]
    and the removed edges are kept in node.meta["collapsed_edges"] so the node map JSON lists them.

    Run it AFTER hide_constants, otherwise constants with many users get this marker
    instead of the usual row.
    """
    name = "collapse_fanout"

    def __init__(self, threshold: int = 3) -> None:
        if threshold < 1:
            raise ValueError("collapse_fanout: threshold must be at least 1")
        self.threshold = threshold

    @classmethod
    def from_options(cls, options: dict[str, list[str]]) -> "CollapseFanoutPass":
        """Option: threshold=3 (a node with MORE outgoing edges than this is collapsed)."""
        unknown = set(options) - {"threshold"}
        if unknown:
            raise ValueError(f"collapse_fanout: unknown option(s) {', '.join(sorted(unknown))}; use threshold")
        if "threshold" not in options:
            return cls()
        try:
            return cls(int(options["threshold"][-1]))
        except ValueError:
            raise ValueError("collapse_fanout: threshold must be an integer") from None

    def run(self, graph: Graph) -> Graph:
        """Remove every outgoing edge of each wide node and leave rows describing them."""
        edges_out = Counter(e.src_id for e in graph.edges)  # one pass: edges leaving each node
        wide = {i for i, count in edges_out.items() if count > self.threshold}

        kept = []
        per_port: Counter = Counter()  # (source id, source port) -> how many edges collapsed
        touched: set[int] = set()
        for e in graph.edges:
            if e.src_id not in wide:
                kept.append(e)
                continue
            graph.nodes[e.src_id].meta.setdefault("collapsed_edges", []).append(e)
            per_port[(e.src_id, e.src_port)] += 1
            text = f"[{e.dst_port}]: from id {e.src_id}"
            if e.dtype is not None or e.shape is not None:  # merged group edges carry no tensor info
                text += f": {e.dtype or '?'}{format_shape(e.shape)}"
            graph.nodes[e.dst_id].meta.setdefault("extra_rows", []).append((e.dst_port, text))
            touched.update((e.src_id, e.dst_id))

        for (src_id, port), n in sorted(per_port.items()):
            graph.nodes[src_id].meta.setdefault("extra_rows", []).append(
                (port, f"out[{port}] -> {n} consumer{'s' if n > 1 else ''} (collapsed)"))

        for i in touched:  # keep rows in port order, like hide_constants
            graph.nodes[i].meta["extra_rows"].sort(key=lambda r: r[0])
        graph.edges = kept
        return graph
