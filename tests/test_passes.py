from ovgraph import Edge, Graph, Node, Pipeline
from ovgraph.passes import HideConstantsPass, HighlightPass, get_pass, available_passes


def test_highlight_by_id_and_type(small_graph):
    out = Pipeline([HighlightPass(node_ids={0}, op_types={"Add"})]).run(small_graph)
    assert out.nodes[0].meta["border_color"] == "red"
    assert out.nodes[3].meta["border_color"] == "red"
    assert "border_color" not in out.nodes[2].meta


def test_hide_constants_removes_nodes_and_edges(small_graph):
    out = Pipeline([HideConstantsPass()]).run(small_graph)
    assert 1 not in out.nodes
    assert all(e.src_id != 1 and e.dst_id != 1 for e in out.edges)
    assert len(out.edges) == 3  # the three non-constant edges remain


def test_hide_constants_rows_keep_port_and_mark_shared(small_graph):
    out = Pipeline([HideConstantsPass()]).run(small_graph)
    assert out.nodes[2].meta["extra_rows"] == [(1, "[1]: f32[768,768] (shared: 1)")]
    assert out.nodes[3].meta["extra_rows"] == [(1, "[1]: f32[768,768] (shared: 1)")]


def test_unshared_constant_has_no_marker():
    g = Graph()
    g.add_node(Node(0, "c", "Constant"))
    g.add_node(Node(1, "n", "Add"))
    g.add_edge(Edge(0, 0, 1, 0, (4,), "i64"))
    out = Pipeline([HideConstantsPass()]).run(g)
    assert out.nodes[1].meta["extra_rows"] == [(0, "[0]: i64[4]")]


def test_original_graph_not_modified(small_graph):
    Pipeline([HideConstantsPass(), HighlightPass(op_types={"Add"})]).run(small_graph)
    assert 1 in small_graph.nodes and len(small_graph.edges) == 5
    assert all(n.meta == {} for n in small_graph.nodes.values())


def test_dynamic_and_missing_metadata():
    g = Graph()
    g.add_node(Node(0, "c", "Constant"))
    g.add_node(Node(1, "n", "Add"))
    g.add_edge(Edge(0, 0, 1, 2, (None, 3), None))
    out = Pipeline([HideConstantsPass()]).run(g)
    assert out.nodes[1].meta["extra_rows"] == [(2, "[2]: ?[?,3]")]


def test_rows_sorted_by_port():
    g = Graph()
    g.add_node(Node(0, "c0", "Constant"))
    g.add_node(Node(1, "c1", "Constant"))
    g.add_node(Node(2, "n", "Select"))
    g.add_edge(Edge(1, 0, 2, 2, (1,), "f32"))
    g.add_edge(Edge(0, 0, 2, 1, (1,), "f32"))
    out = Pipeline([HideConstantsPass()]).run(g)
    assert [p for p, _ in out.nodes[2].meta["extra_rows"]] == [1, 2]


def test_registry():
    assert {"highlight", "hide_constants"} <= set(available_passes())
    assert get_pass("highlight") is HighlightPass


def test_fold_constant_converts_rewires_and_keeps_activation_converts():
    from ovgraph import Edge, Graph, Node
    from ovgraph.passes.fold_constant_converts import FoldConstantConvertsPass
    from ovgraph.passes.hide_constants import HideConstantsPass

    g = Graph()
    for i, op in enumerate(["Parameter", "Constant", "Convert", "MatMul", "Convert", "Result"]):
        g.add_node(Node(i, f"n{i}", op))
    g.add_edge(Edge(1, 0, 2, 0, (4, 4), "f16"))   # weight const -> Convert
    g.add_edge(Edge(2, 0, 3, 1, (4, 4), "f32"))   # Convert -> MatMul port 1
    g.add_edge(Edge(0, 0, 3, 0, (1, 4), "f32"))
    g.add_edge(Edge(3, 0, 4, 0, (1, 4), "f32"))   # activation -> Convert (must stay)
    g.add_edge(Edge(4, 0, 5, 0, (1, 4), "f16"))

    out = FoldConstantConvertsPass().run(g.clone())
    assert 2 not in out.nodes and 4 in out.nodes
    assert Edge(1, 0, 3, 1, (4, 4), "f32") in out.edges   # constant now feeds MatMul directly, as f32
    assert len(g) == 6                                      # input untouched

    hidden = HideConstantsPass().run(out)
    assert hidden.nodes[3].meta["extra_rows"] == [(1, "[1]: f32[4,4]")]
