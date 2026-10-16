from __future__ import annotations

from ..graph import Edge, Graph
from .base import Pass, register_pass


@register_pass
class FoldConstantConvertsPass(Pass):
    """Custom pass for FP16-exported models (e.g. TinyLlama).

    Weights are stored as f16 Constants, and every use is wrapped as
        Constant (f16) -> Convert (to f32) -> consumer
    so after `hide_constants` the picture is still full of Convert boxes that
    carry no information. This pass removes a Convert whose ONLY input is a
    Constant and connects the Constant straight to the Convert's consumers,
    using the Convert's output dtype (f32). Run it BEFORE hide_constants so the
    constant then shows up as a normal row on its consumer:
        [1]: f32[2048,2048]
    A Convert fed by anything else (a real activation cast) is left alone.
    """
    name = "fold_constant_converts"

    def run(self, graph: Graph) -> Graph:
        node_type = {i: n.op_type for i, n in graph.nodes.items()}
        # Convert id -> its single incoming edge (only if that edge comes from a Constant)
        feeders: dict[int, Edge] = {}
        in_count: dict[int, int] = {}
        for e in graph.edges:
            if node_type[e.dst_id] == "Convert":
                in_count[e.dst_id] = in_count.get(e.dst_id, 0) + 1
                if node_type[e.src_id] == "Constant":
                    feeders[e.dst_id] = e
        foldable = {c: e for c, e in feeders.items() if in_count[c] == 1}

        new_edges: list[Edge] = []
        for e in graph.edges:
            if e.dst_id in foldable:
                continue  # Constant -> Convert edge disappears
            if e.src_id in foldable:  # Convert -> consumer becomes Constant -> consumer
                src = foldable[e.src_id]
                new_edges.append(Edge(src.src_id, src.src_port, e.dst_id, e.dst_port, e.shape, e.dtype))
            else:
                new_edges.append(e)

        for cid in foldable:
            del graph.nodes[cid]
        graph.edges = new_edges
        return graph
