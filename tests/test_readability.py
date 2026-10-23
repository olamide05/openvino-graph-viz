import json

from ovgraph import Edge, Graph, Node, Pipeline
from ovgraph.cli import build_parser, build_pipeline
from ovgraph.export import node_map, write_node_map
from ovgraph.passes import CollapseFanoutPass, CollapseModulesPass, HideConstantsPass, get_pass
from ovgraph.render import GraphvizRenderer


def fan_graph(consumers: int = 5) -> Graph:
    """Parameter -> `consumers` Add nodes, each Add -> its own Result."""
    g = Graph()
    g.add_node(Node(0, "mask", "Parameter"))
    for i in range(consumers):
        g.add_node(Node(1 + i, f"add{i}", "Add"))
        g.add_edge(Edge(0, 0, 1 + i, 1, (1, 4), "f32"))
    return g


# ---------- hide names + node map ----------

def test_hide_names_leaves_name_out_but_keeps_op_and_id(small_graph):
    shown = GraphvizRenderer().to_dot(small_graph)
    hidden = GraphvizRenderer(show_names=False).to_dot(small_graph)
    assert "<B>mm</B>" in shown
    assert "<B>mm</B>" not in hidden and "<B>MatMul</B> (id 2)" in hidden


def test_node_map_has_name_op_attributes_and_rows(small_graph):
    small_graph.nodes[2].attributes["transpose_b"] = "true"
    out = Pipeline([HideConstantsPass()]).run(small_graph)
    m = node_map(out)
    assert m["2"]["name"] == "mm" and m["2"]["op_type"] == "MatMul"
    assert m["2"]["attributes"] == {"transpose_b": "true"}
    assert m["2"]["rows"] == ["[1]: f32[768,768] (shared: 1)"]


def test_write_node_map_is_valid_json(tmp_path, small_graph):
    path = tmp_path / "map.json"
    write_node_map(small_graph, path)
    assert json.loads(path.read_text())["0"]["name"] == "input"


# ---------- collapse_fanout ----------

def test_fanout_removes_edges_of_wide_node():
    out = Pipeline([CollapseFanoutPass(threshold=3)]).run(fan_graph(5))
    assert out.edges == []
    assert any("out[0] -> 5 consumers" in text for _p, text in out.nodes[0].meta["extra_rows"])


def test_fanout_keeps_connectivity_in_consumer_rows_and_map():
    out = Pipeline([CollapseFanoutPass(threshold=3)]).run(fan_graph(5))
    assert out.nodes[3].meta["extra_rows"] == [(1, "[1]: from id 0: f32[1,4]")]
    collapsed = node_map(out)["0"]["collapsed_outputs"]
    assert sorted(c["dst_id"] for c in collapsed) == [1, 2, 3, 4, 5]
    assert all(c["dst_port"] == 1 and c["dtype"] == "f32" for c in collapsed)


def test_fanout_row_has_no_tensor_text_when_edge_has_none():
    g = Graph()
    g.add_node(Node(0, "src", "Add"))
    for i in range(4):
        g.add_node(Node(1 + i, f"c{i}", "Add"))
        g.add_edge(Edge(0, 0, 1 + i, 0))  # no dtype, no shape
    out = Pipeline([CollapseFanoutPass()]).run(g)
    assert out.nodes[1].meta["extra_rows"] == [(0, "[0]: from id 0")]


def test_fanout_leaves_narrow_nodes_alone():
    out = Pipeline([CollapseFanoutPass(threshold=3)]).run(fan_graph(3))  # exactly 3: not MORE than 3
    assert len(out.edges) == 3 and "extra_rows" not in out.nodes[0].meta


def test_fanout_threshold_option_and_errors():
    assert get_pass("collapse-fanout").from_options({"threshold": ["7"]}).threshold == 7
    for bad in ({"threshold": ["x"]}, {"nope": ["1"]}):
        try:
            get_pass("collapse_fanout").from_options(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)


def test_fanout_does_not_modify_input():
    g = fan_graph(5)
    Pipeline([CollapseFanoutPass()]).run(g)
    assert len(g.edges) == 5 and g.nodes[0].meta == {}


def test_fanout_counts_each_output_port():
    g = fan_graph(4)
    g.add_node(Node(9, "r", "Result"))
    g.add_edge(Edge(0, 1, 9, 0, (1,), "i64"))  # a second output port with one consumer
    out = Pipeline([CollapseFanoutPass()]).run(g)
    texts = [t for _p, t in out.nodes[0].meta["extra_rows"]]
    assert "out[0] -> 4 consumers (collapsed)" in texts and "out[1] -> 1 consumer (collapsed)" in texts


# ---------- collapse_modules ----------

def layered_graph() -> Graph:
    """input -> layers.0 (a -> b) -> layers.1 (c -> d) -> output, plus input -> layers.1 directly."""
    g = Graph()
    g.add_node(Node(0, "input", "Parameter"))
    g.add_node(Node(1, "__module.model.layers.0.attn/a", "MatMul"))
    g.add_node(Node(2, "__module.model.layers.0.mlp/b", "Add"))
    g.add_node(Node(3, "__module.model.layers.1.attn/c", "MatMul"))
    g.add_node(Node(4, "__module.model.layers.1.mlp/d", "Add"))
    g.add_node(Node(5, "output", "Result"))
    for e in [Edge(0, 0, 1, 0), Edge(1, 0, 2, 0), Edge(2, 0, 3, 0), Edge(3, 0, 4, 0),
              Edge(0, 0, 3, 1), Edge(4, 0, 5, 0)]:
        g.add_edge(e)
    return g


def test_modules_become_one_box_each():
    out = Pipeline([CollapseModulesPass()]).run(layered_graph())
    names = sorted(n.name for n in out.nodes.values())
    assert names == ["input", "layers.0", "layers.1", "output"]
    groups = [n for n in out.nodes.values() if n.op_type == "Group"]
    assert [n.meta["extra_rows"] for n in groups] == [[(0, "2 nodes")]] * 2


def test_module_edges_inside_vanish_and_between_merge_with_count():
    out = Pipeline([CollapseModulesPass()]).run(layered_graph())
    by_name = {n.name: n.id for n in out.nodes.values()}
    pairs = {(e.src_id, e.dst_id): e.label for e in out.edges}
    assert pairs[(by_name["input"], by_name["layers.0"])] == "1 edge"
    assert pairs[(by_name["layers.0"], by_name["layers.1"])] == "1 edge"
    assert pairs[(by_name["input"], by_name["layers.1"])] == "1 edge"
    assert pairs[(by_name["layers.1"], by_name["output"])] == "1 edge"
    assert len(out.edges) == 4  # the two edges inside a layer are gone


def test_many_edges_between_modules_merge_into_one_labelled_edge():
    g = layered_graph()
    g.add_edge(Edge(1, 1, 3, 2))  # a second layers.0 -> layers.1 edge
    out = Pipeline([CollapseModulesPass()]).run(g)
    by_name = {n.name: n.id for n in out.nodes.values()}
    labels = [e.label for e in out.edges if (e.src_id, e.dst_id) == (by_name["layers.0"], by_name["layers.1"])]
    assert labels == ["2 edges"]


def test_expand_keeps_one_module_unfolded():
    out = Pipeline([CollapseModulesPass(expand=["layers.0"])]).run(layered_graph())
    names = sorted(n.name for n in out.nodes.values())
    assert "layers.1" in names and "layers.0" not in names
    assert "__module.model.layers.0.attn/a" in names


def test_members_are_kept_for_the_node_map():
    out = Pipeline([CollapseModulesPass()]).run(layered_graph())
    group = next(n for n in out.nodes.values() if n.name == "layers.1")
    members = node_map(out)[str(group.id)]["members"]
    assert [m["id"] for m in members] == [3, 4] and members[0]["op_type"] == "MatMul"


def test_pattern_needs_a_group_and_must_be_valid():
    for bad in ({"pattern": ["layers"]}, {"pattern": ["("]}):
        try:
            CollapseModulesPass.from_options(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)


def test_modules_then_fanout_together():
    """The overview recipe: fold layers, then collapse the shared input that feeds every layer."""
    g = Graph()
    g.add_node(Node(0, "mask", "Parameter"))
    for i in range(5):
        g.add_node(Node(1 + i, f"__module.model.layers.{i}.attn/x", "Add"))
        g.add_edge(Edge(0, 0, 1 + i, 1, (1, 4), "f32"))
    out = Pipeline([CollapseModulesPass(), CollapseFanoutPass()]).run(g)
    assert out.edges == [] and len(out.nodes) == 6  # mask + 5 layer boxes
    assert any("5 consumers" in t for _p, t in out.nodes[0].meta["extra_rows"])


def test_edge_label_override_is_drawn():
    g = layered_graph()
    out = Pipeline([CollapseModulesPass()]).run(g)
    assert "1 edge" in GraphvizRenderer().to_dot(out)


# ---------- CLI ----------

def test_cli_flags_build_passes_in_order():
    args = build_parser().parse_args(["m.xml", "--hide-constants", "--collapse-modules", "--collapse-fanout",
                                      "--hide-names", "--node-map", "out.json"])
    assert [p.name for p in build_pipeline(args).passes] == ["hide_constants", "collapse_modules", "collapse_fanout"]
    assert args.hide_names and args.node_map == "out.json"
