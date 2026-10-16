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


# ---- assignment requirements: highlight by NAME, pass order, enable/disable, options ----

def test_highlight_by_names_only_matching_nodes(small_graph):
    small_graph.add_node(Node(9, "mm", "Relu"))  # duplicate name "mm": both must light up
    out = Pipeline([HighlightPass(names=["mm", "out"])]).run(small_graph)
    assert {i for i, n in out.nodes.items() if "border_color" in n.meta} == {2, 4, 9}
    assert "border_color" not in out.nodes[0].meta and "border_color" not in out.nodes[3].meta
    assert "border_color" not in small_graph.nodes[2].meta      # original untouched


def test_passes_run_in_the_given_order():
    from ovgraph.passes import Pass
    calls = []

    class Rec(Pass):
        def __init__(self, tag):
            self.tag = tag

        def run(self, graph):
            calls.append(self.tag)
            return graph

    g = Graph()
    g.add_node(Node(0, "a", "Add"))
    Pipeline([Rec("first"), Rec("second"), Rec("third")]).run(g)
    assert calls == ["first", "second", "third"]


def test_order_changes_the_result():
    """fold_constant_converts only works BEFORE hide_constants."""
    from ovgraph.passes import FoldConstantConvertsPass
    g = Graph()
    for i, op in enumerate(["Constant", "Convert", "MatMul"]):
        g.add_node(Node(i, f"n{i}", op))
    g.add_edge(Edge(0, 0, 1, 0, (4, 4), "f16"))
    g.add_edge(Edge(1, 0, 2, 1, (4, 4), "f32"))
    good = Pipeline([FoldConstantConvertsPass(), HideConstantsPass()]).run(g)
    bad = Pipeline([HideConstantsPass(), FoldConstantConvertsPass()]).run(g)
    assert 1 not in good.nodes
    assert 1 in bad.nodes


def test_disabled_pass_is_skipped_and_can_be_reenabled(small_graph):
    pipe = Pipeline([HideConstantsPass(), HighlightPass(names=["mm"])])
    pipe.set_enabled("hide_constants", False)
    out = pipe.run(small_graph)
    assert 1 in out.nodes and out.nodes[2].meta["border_color"] == "red"
    pipe.set_enabled("hide_constants", True)
    assert 1 not in pipe.run(small_graph).nodes


def test_pass_options_and_hyphenated_names():
    p = get_pass("hide-constants")
    assert p is HideConstantsPass
    h = HighlightPass.from_options({"names": ["a", "b"], "ids": ["3"], "ops": ["MatMul"], "color": ["blue"]})
    assert (h.names, h.node_ids, h.op_types, h.color) == ({"a", "b"}, {3}, {"MatMul"}, "blue")
    import pytest
    with pytest.raises(ValueError):
        HighlightPass.from_options({"nope": ["x"]})
    with pytest.raises(ValueError):
        HideConstantsPass.from_options({"color": ["red"]})
    with pytest.raises(KeyError):
        get_pass("does-not-exist")


def test_constant_feeding_constant_is_kept_while_it_has_a_visible_connection():
    g = Graph()
    g.add_node(Node(0, "a", "Constant"))
    g.add_node(Node(1, "b", "Constant"))
    g.add_node(Node(2, "m", "MatMul"))
    g.add_edge(Edge(0, 0, 1, 0, (2,), "f32"))
    g.add_edge(Edge(1, 0, 2, 1, (2,), "f32"))
    out = Pipeline([HideConstantsPass()]).run(g)
    assert 0 in out.nodes and 1 in out.nodes          # a -> b is still a visible connection, so both stay
    assert out.nodes[2].meta["extra_rows"] == [(1, "[1]: f32[2]")]   # b's use by the MatMul became a row
