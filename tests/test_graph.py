import pytest
from ovgraph import Edge, Graph, Node, format_shape


def test_duplicate_names_allowed_ids_unique():
    g = Graph()
    g.add_node(Node(0, "same", "Add"))
    g.add_node(Node(1, "same", "Add"))
    with pytest.raises(ValueError):
        g.add_node(Node(1, "other", "Add"))


def test_multiple_edges_between_same_pair():
    g = Graph()
    g.add_node(Node(0, "a", "Split"))
    g.add_node(Node(1, "b", "Concat"))
    g.add_edge(Edge(0, 0, 1, 0))
    g.add_edge(Edge(0, 1, 1, 1))
    assert len(g.out_edges(0)) == 2


def test_edge_to_unknown_node_fails():
    g = Graph()
    g.add_node(Node(0, "a", "Add"))
    with pytest.raises(KeyError):
        g.add_edge(Edge(0, 0, 9, 0))


def test_clone_is_independent(small_graph):
    c = small_graph.clone()
    c.nodes[2].meta["x"] = 1
    c.nodes[2].attributes["k"] = "v"
    assert small_graph.nodes[2].meta == {} and small_graph.nodes[2].attributes == {}


def test_format_shape_dynamic_and_missing():
    assert format_shape((1, None, 768)) == "[1,?,768]"
    assert format_shape(None) == "?"
    assert format_shape(()) == "[]"
