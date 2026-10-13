"""A 3-box demo. Run it:  python examples/demo.py   then open demo_before.svg and demo_after.svg"""
from ovgraph import Edge, Graph, Node, Pipeline
from ovgraph.passes import HideConstantsPass, HighlightPass
from ovgraph.render import GraphvizRenderer

# 1. Build a tiny model by hand: input -> MatMul <- weights
g = Graph()
g.add_node(Node(0, "input", "Parameter"))
g.add_node(Node(1, "weights", "Constant"))
g.add_node(Node(2, "mm", "MatMul"))
g.add_edge(Edge(0, 0, 2, 0, shape=(1, 768), dtype="f32"))     # box 0 -> box 2, slot 0
g.add_edge(Edge(1, 0, 2, 1, shape=(768, 768), dtype="f32"))   # box 1 -> box 2, slot 1


def show(title, graph):
    print(f"\n--- {title} ---")
    for n in graph.nodes.values():
        print(f"  box {n.id}: {n.name} ({n.op_type})  notes={n.meta}")
    for e in graph.edges:
        print(f"  arrow: box {e.src_id} --> box {e.dst_id} (slot {e.dst_port})")


show("BEFORE", g)

# 2. Run the tidy-up steps. This makes a COPY; g itself is not touched.
tidy = Pipeline([HideConstantsPass(), HighlightPass(op_types={"MatMul"})]).run(g)
show("AFTER", tidy)

# 3. Draw both
GraphvizRenderer().render_svg(g, "demo_before.svg")
GraphvizRenderer().render_svg(tidy, "demo_after.svg")
print("\nwrote demo_before.svg and demo_after.svg")
print("original untouched? weights still there:", 1 in g.nodes)
