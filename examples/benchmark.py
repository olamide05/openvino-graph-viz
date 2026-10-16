"""Time the pipeline on synthetic graphs of increasing size (default up to ~5k nodes).

Run:  python examples/benchmark.py [sizes...]      e.g.  python examples/benchmark.py 1000 5000

Graph shape: a chain of "layers". Each layer = Parameter-like MatMul fed by the previous layer
and one Constant (weights), plus an Add with a shared Constant (bias). So roughly 40% of nodes
are Constants, similar to a real exported model.
"""
import sys
import time

from ovgraph import Edge, Graph, Node, Pipeline
from ovgraph.passes.hide_constants import HideConstantsPass
from ovgraph.passes.highlight import HighlightPass
from ovgraph.render.graphviz_renderer import GraphvizRenderer


def make_graph(n_nodes: int) -> Graph:
    g = Graph()
    nid = 0

    def node(op, name=None):
        nonlocal nid
        g.add_node(Node(id=nid, name=name or f"{op}_{nid}", op_type=op))
        nid += 1
        return nid - 1

    shape, dt = (1, 128, 64), "f32"
    prev = node("Parameter", "input")
    shared_bias = node("Constant", "shared_bias")
    while nid < n_nodes - 2:
        w = node("Constant")
        mm = node("MatMul")
        add = node("Add")
        g.add_edge(Edge(prev, 0, mm, 0, shape, dt))
        g.add_edge(Edge(w, 0, mm, 1, (64, 64), dt))
        g.add_edge(Edge(mm, 0, add, 0, shape, dt))
        g.add_edge(Edge(shared_bias, 0, add, 1, (64,), dt))
        prev = add
    res = node("Result", "output")
    g.add_edge(Edge(prev, 0, res, 0, shape, dt))
    return g


def main(sizes):
    print(f"{'nodes':>7} {'edges':>7} | {'passes (s)':>10} | {'drawn':>6} | {'render fast (s)':>15}")
    for n in sizes:
        g = make_graph(n)
        t0 = time.perf_counter()
        out = Pipeline([HideConstantsPass(), HighlightPass(op_types=["MatMul"])]).run(g)
        t_pass = time.perf_counter() - t0
        t0 = time.perf_counter()
        GraphvizRenderer(fast=True).render_svg(out)
        t_render = time.perf_counter() - t0
        print(f"{len(g):>7} {len(g.edges):>7} | {t_pass:>10.3f} | {len(out):>6} | {t_render:>15.2f}")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [500, 1000, 2500, 5000])
