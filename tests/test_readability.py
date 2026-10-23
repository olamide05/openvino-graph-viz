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


def test_single_node_group_is_not_plural():
    g = Graph()
    g.add_node(Node(0, "m.layers.0.x", "Add"))
    out = Pipeline([CollapseModulesPass()]).run(g)
    assert out.nodes[1].meta["extra_rows"] == [(0, "1 node")]


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


# ---------- review fixes ----------

def test_folded_edges_are_recoverable_from_the_node_map():
    out = Pipeline([CollapseModulesPass()]).run(layered_graph())
    m = node_map(out)
    by_name = {n.name: str(n.id) for n in out.nodes.values()}
    edges = {(e["src_id"], e["dst_id"]) for e in m[by_name["layers.0"]]["folded_edges"]}
    assert edges == {(0, 1), (1, 2), (2, 3)}  # into the box, inside it, and out of it


def test_fanout_before_modules_keeps_collapsed_edges():
    out = Pipeline([CollapseFanoutPass(threshold=2), CollapseModulesPass()]).run(fan_graph_in_layers())
    group = next(n for n in out.nodes.values() if n.op_type == "Group")
    assert node_map(out)[str(group.id)]["collapsed_outputs"] != []


def fan_graph_in_layers() -> Graph:
    g = Graph()
    g.add_node(Node(0, "m.layers.0.src", "Add"))
    for i in range(4):
        g.add_node(Node(1 + i, f"m.layers.1.c{i}", "Add"))
        g.add_edge(Edge(0, 0, 1 + i, 0, (1,), "f32"))
    return g


def test_fanout_after_modules_names_the_merged_edge_count():
    g = Graph()
    g.add_node(Node(0, "m.layers.0.src", "Add"))
    for i in range(1, 6):
        g.add_node(Node(i, f"m.layers.{i}.c", "Add"))
        g.add_edge(Edge(0, 0, i, 0, (1,), "f32"))
        g.add_edge(Edge(0, 1, i, 1, (1,), "f32"))  # two edges per pair, merged into "2 edges"
    out = Pipeline([CollapseModulesPass(), CollapseFanoutPass()]).run(g)
    rows = [t for n in out.nodes.values() for _p, t in n.meta.get("extra_rows", [])]
    assert any(t == "shared inputs: id 6" for t in rows)  # the boxes get one short row; details stay in the JSON


def test_threshold_zero_gets_its_own_error_message():
    try:
        get_pass("collapse_fanout").from_options({"threshold": ["0"]})
    except ValueError as e:
        assert "at least 1" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_every_original_edge_is_recoverable_after_both_passes():
    """Drawn edges + folded edges + collapsed edges must add up to exactly the original edges."""
    g = Graph()
    g.add_node(Node(0, "mask", "Parameter"))
    for layer in range(4):
        a, b = 1 + 2 * layer, 2 + 2 * layer
        g.add_node(Node(a, f"m.layers.{layer}.attn", "MatMul"))
        g.add_node(Node(b, f"m.layers.{layer}.mlp", "Add"))
        g.add_edge(Edge(0, 0, a, 1, (1,), "f32"))  # shared input feeds every layer
        g.add_edge(Edge(a, 0, b, 0, (1,), "f32"))  # edge inside a layer
        if layer:
            g.add_edge(Edge(b - 2, 0, a, 0, (1,), "f32"))  # previous layer -> this layer
    original = sorted((e.src_id, e.src_port, e.dst_id, e.dst_port) for e in g.edges)
    for passes in ([CollapseModulesPass(), CollapseFanoutPass(2)], [CollapseFanoutPass(2), CollapseModulesPass()]):
        out = Pipeline(passes).run(g)
        found = [(e.src_id, e.src_port, e.dst_id, e.dst_port) for e in out.edges if e.label is None]
        for n in out.nodes.values():
            found += [(e.src_id, e.src_port, e.dst_id, e.dst_port) for e in n.meta.get("folded_edges", [])]
            found += [(e.src_id, e.src_port, e.dst_id, e.dst_port)
                      for e in n.meta.get("collapsed_edges", []) if e.label is None]
        assert sorted(found) == original


# ---------- absorbing generated-name nodes (KV-cache plumbing) ----------

def kv_graph() -> Graph:
    """ReadValue_1 -> Gather_2 -> layers.0 node -> Assign_3, plus a named embedding that feeds only layers.0."""
    g = Graph()
    g.add_node(Node(0, "__module.model.embed_tokens/Gather", "Gather"))  # named, but not in a layer
    g.add_node(Node(1, "ReadValue_1", "ReadValue"))
    g.add_node(Node(2, "Gather_2", "Gather"))
    g.add_node(Node(3, "__module.model.layers.0.attn/MatMul", "MatMul"))
    g.add_node(Node(4, "Assign_3", "Assign"))
    for e in [Edge(1, 0, 2, 0), Edge(2, 0, 3, 0), Edge(0, 0, 3, 1), Edge(3, 0, 4, 0)]:
        g.add_edge(e)
    return g


def test_generated_name_chain_joins_its_layer():
    out = Pipeline([CollapseModulesPass()]).run(kv_graph())
    group = next(n for n in out.nodes.values() if n.op_type == "Group")
    assert sorted(m["id"] for m in group.meta["members"]) == [1, 2, 3, 4]  # feeder chain, layer node, sink
    assert [n.name for n in out.nodes.values() if n.op_type != "Group"] == ["__module.model.embed_tokens/Gather"]


def test_named_nodes_are_never_moved_by_absorb():
    out = Pipeline([CollapseModulesPass()]).run(kv_graph())
    group = next(n for n in out.nodes.values() if n.op_type == "Group")
    assert 0 not in [m["id"] for m in group.meta["members"]]  # the embedding only feeds layers.0, but has a real name


def test_shared_generated_node_goes_into_the_shared_box():
    g = Graph()
    g.add_node(Node(0, "Gather_9", "Gather"))
    for i in range(2):
        g.add_node(Node(1 + i, f"m.layers.{i}.x", "Add"))
        g.add_edge(Edge(0, 0, 1 + i, 0))  # feeds two different layers
    out = Pipeline([CollapseModulesPass()]).run(g)
    assert all(n.op_type == "Group" for n in out.nodes.values())
    shared = next(n for n in out.nodes.values() if n.name.startswith("shared"))
    assert [m["name"] for m in shared.meta["members"]] == ["Gather_9"]


def test_shared_generated_node_stays_outside_when_grouping_is_off():
    g = Graph()
    g.add_node(Node(0, "Gather_9", "Gather"))
    for i in range(2):
        g.add_node(Node(1 + i, f"m.layers.{i}.x", "Add"))
        g.add_edge(Edge(0, 0, 1 + i, 0))
    out = Pipeline([CollapseModulesPass(group_shared=False)]).run(g)
    assert [n.name for n in out.nodes.values() if n.op_type != "Group"] == ["Gather_9"]


def test_absorb_can_be_switched_off():
    out = Pipeline([CollapseModulesPass(absorb=False)]).run(kv_graph())
    assert sorted(n.name for n in out.nodes.values() if n.op_type != "Group") == [
        "Assign_3", "Gather_2", "ReadValue_1", "__module.model.embed_tokens/Gather"]
    assert CollapseModulesPass.from_options({"absorb": ["false"]}).absorb is False
    try:
        CollapseModulesPass.from_options({"absorb": ["maybe"]})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_expanded_layer_keeps_its_plumbing_as_separate_nodes():
    out = Pipeline([CollapseModulesPass(expand=["layers.0"])]).run(kv_graph())
    assert not [n for n in out.nodes.values() if n.op_type == "Group"] and len(out.nodes) == 5


def test_absorb_loses_no_edges():
    g = kv_graph()
    original = sorted((e.src_id, e.dst_id) for e in g.edges)
    out = Pipeline([CollapseModulesPass()]).run(g)
    found = [(e.src_id, e.dst_id) for e in out.edges if e.label is None]
    for n in out.nodes.values():
        found += [(e.src_id, e.dst_id) for e in n.meta.get("folded_edges", [])]
    assert sorted(found) == original
