import shutil

import pytest
from ovgraph import Edge, Graph, Node, Pipeline
from ovgraph.passes import HideConstantsPass, HighlightPass
from ovgraph.render import GraphvizRenderer

needs_dot = pytest.mark.skipif(shutil.which("dot") is None, reason="Graphviz 'dot' not installed")


def test_dot_has_ports_and_edge_ports(small_graph):
    src = GraphvizRenderer().to_dot(small_graph)
    assert 'PORT="in1"' in src and 'PORT="out0"' in src
    assert "2:out0:s -> 3:in0:n" in src.replace('"', "")


def test_highlight_becomes_red_border(small_graph):
    g = Pipeline([HighlightPass(node_ids={2})]).run(small_graph)
    src = GraphvizRenderer().to_dot(g)
    assert "color=red" in src


def test_constant_rows_rendered_and_no_constant_node(small_graph):
    g = Pipeline([HideConstantsPass()]).run(small_graph)
    src = GraphvizRenderer().to_dot(g)
    assert "[1]: f32[768,768] (shared: 1)" in src
    assert "(id 1)" not in src  # constant node itself is gone


def test_names_are_html_escaped():
    g = Graph()
    g.add_node(Node(0, "a<b>&c", "Add"))
    assert "a&lt;b&gt;&amp;c" in GraphvizRenderer().to_dot(g)


def test_duplicate_names_get_separate_nodes():
    g = Graph()
    g.add_node(Node(0, "dup", "Add"))
    g.add_node(Node(1, "dup", "Add"))
    src = GraphvizRenderer().to_dot(g)
    assert src.count("dup") == 2


def test_node_without_edges_renders():
    g = Graph()
    g.add_node(Node(0, "lonely", "Parameter"))
    assert "lonely" in GraphvizRenderer().to_dot(g)


@needs_dot
def test_svg_output(small_graph, tmp_path):
    out = tmp_path / "g.svg"
    svg = GraphvizRenderer().render_svg(small_graph, out)
    assert svg.lstrip().startswith(b"<?xml") and b"<svg" in svg
    assert out.exists()


def _chain(n):
    g = Graph()
    for i in range(n):
        g.add_node(Node(i, f"n{i}", "Add"))
    for i in range(n - 1):
        g.add_edge(Edge(i, 0, i + 1, 0))
    return g


def test_small_graph_keeps_default_layout():
    assert "splines" not in GraphvizRenderer().to_dot(_chain(10))


def test_big_graph_switches_to_fast_layout_automatically():
    src = GraphvizRenderer().to_dot(_chain(500))
    assert "splines=line" in src and "nslimit=1" in src


def test_fast_flag_overrides_size():
    assert "splines=line" in GraphvizRenderer(fast=True).to_dot(_chain(5))
    assert "splines" not in GraphvizRenderer(fast=False).to_dot(_chain(500))
