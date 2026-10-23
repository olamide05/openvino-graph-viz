"""Pass that folds all nodes of one module (for example one transformer layer) into a single box.
"""
from __future__ import annotations

import re
from collections import Counter

from ..graph import Edge, Graph, Node
from .base import Pass, register_pass

GROUP = "Group"


@register_pass
class CollapseModulesPass(Pass):
    """Draw one box per module instead of every node inside it.

    The module of a node comes from its NAME: the first capture group of `pattern`.
    The default `(layers\\.\\d+)` turns
        __module.model.layers.7.self_attn/aten::unsqueeze/Unsqueeze   ->  "layers.7"
    so a 22-layer model becomes 22 boxes. Nodes whose name does not match stay as they are.

    - Edges inside a module disappear; edges between boxes are merged into one edge
      labelled with how many edges it stands for ("17 edges").
    - `expand` lists modules to leave unfolded, to look inside one layer:
          collapse_modules:expand=layers.0
    - The folded nodes (id, name, op type, attributes, constant rows) are kept in node.meta["members"],
      and every edge that is no longer drawn as itself in node.meta["folded_edges"], so the node map JSON
      still lists them. Run collapse_fanout AFTER this pass (the other order also works, but the rows are clearer).
    """
    name = "collapse_modules"

    def __init__(self, pattern: str = r"(layers\.\d+)", expand: list[str] | set[str] | None = None) -> None:
        self.regex = re.compile(pattern)
        if self.regex.groups < 1:
            raise ValueError("collapse_modules: pattern needs one capture group, e.g. (layers\\.\\d+)")
        self.expand = set(expand or ())

    @classmethod
    def from_options(cls, options: dict[str, list[str]]) -> "CollapseModulesPass":
        """Options: pattern=<regex with one group>  expand=layers.0,layers.1"""
        unknown = set(options) - {"pattern", "expand"}
        if unknown:
            raise ValueError(f"collapse_modules: unknown option(s) {', '.join(sorted(unknown))}; use pattern, expand")
        kwargs = {}
        if "pattern" in options:
            kwargs["pattern"] = ",".join(options["pattern"])  # a regex may contain commas, which the option parser split
        if "expand" in options:
            kwargs["expand"] = options["expand"]
        try:
            return cls(**kwargs)
        except re.error as e:
            raise ValueError(f"collapse_modules: bad pattern: {e}") from None

    def _module_of(self, node: Node) -> str | None:
        m = self.regex.search(node.name)
        return m.group(1) if m else None

    def run(self, graph: Graph) -> Graph:
        """Add one Group node per module, rewire the edges to it, then remove the folded nodes."""
        members: dict[str, list[Node]] = {}  # module -> its nodes, in order of first appearance
        for n in graph.nodes.values():
            module = self._module_of(n)
            if module is not None and module not in self.expand:
                members.setdefault(module, []).append(n)

        next_id = max(graph.nodes, default=-1) + 1
        group_of: dict[int, int] = {}  # folded node id -> id of its group box
        for module, nodes in members.items():
            group = Node(next_id, module, GROUP, {}, {
                "members": [{"id": n.id, "name": n.name, "op_type": n.op_type, "attributes": n.attributes,
                             "rows": [text for _port, text in n.meta.get("extra_rows", [])]} for n in nodes],
                "extra_rows": [(0, f"{len(nodes)} node{'s' if len(nodes) > 1 else ''}")],
                # edges collapse_fanout already removed from the members (if it ran first) stay recoverable
                "collapsed_edges": [e for n in nodes for e in n.meta.get("collapsed_edges", [])],
            })
            graph.add_node(group)
            for n in nodes:
                group_of[n.id] = group.id
            next_id += 1

        kept: list[Edge] = []
        merged: Counter = Counter()  # (source box, destination box) -> how many edges merged into it
        for e in graph.edges:
            src = group_of.get(e.src_id, e.src_id)
            dst = group_of.get(e.dst_id, e.dst_id)
            if src == e.src_id and dst == e.dst_id:
                kept.append(e)  # neither end was folded: draw as before
                continue
            # Every edge that is not drawn as itself is recorded on a group (the source's, else the
            # destination's), so the node map keeps the real node ids, ports and tensor info.
            owner = graph.nodes[src] if e.src_id in group_of else graph.nodes[dst]
            owner.meta.setdefault("folded_edges", []).append(e)
            if src != dst:
                merged[(src, dst)] += 1
            # else: both ends are inside the same module; the edge is hidden
        for (src, dst), count in merged.items():
            kept.append(Edge(src, 0, dst, 0, label=f"{count} edge" + ("s" if count > 1 else "")))

        for node_id in group_of:
            del graph.nodes[node_id]
        graph.edges = kept
        return graph
