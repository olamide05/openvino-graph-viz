import pytest
from ovgraph import Edge, Graph, Node


@pytest.fixture
def small_graph():
    """Parameter -> MatMul <- Constant(shared) ; Constant(shared) -> Add ; MatMul -> Add -> Result"""
    g = Graph()
    g.add_node(Node(0, "input", "Parameter"))
    g.add_node(Node(1, "w", "Constant"))
    g.add_node(Node(2, "mm", "MatMul"))
    g.add_node(Node(3, "add", "Add"))
    g.add_node(Node(4, "out", "Result"))
    g.add_edge(Edge(0, 0, 2, 0, (1, 768), "f32"))
    g.add_edge(Edge(1, 0, 2, 1, (768, 768), "f32"))
    g.add_edge(Edge(2, 0, 3, 0, (1, 768), "f32"))
    g.add_edge(Edge(1, 0, 3, 1, (768, 768), "f32"))
    g.add_edge(Edge(3, 0, 4, 0, (1, 768), "f32"))
    return g
