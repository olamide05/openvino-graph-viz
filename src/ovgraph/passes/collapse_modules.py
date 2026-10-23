"""Pass that folds all nodes of one module (for example one transformer layer) into a single box.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from ..graph import Edge, Graph, Node
from .base import Pass, register_pass

GROUP = "Group"
# OpenVINO's generated names look like Gather_70871 (op type, underscore, number). Such a name says
# nothing about which module the node belongs to, unlike __module.model.layers.7.self_attn/...
AUTO_NAME = re.compile(r"^[A-Za-z0-9]+_\d+$")


@register_pass
class CollapseModulesPass(Pass):
    """Draw one box per module instead of every node inside it.

    The module of a node comes from its NAME: the first capture group of `pattern`.
    The default `(layers\\.\\d+)` turns
        __module.model.layers.7.self_attn/aten::unsqueeze/Unsqueeze   ->  "layers.7"
    so a 22-layer model becomes 22 boxes. Nodes whose name does not match stay as they are.

    - Nodes with a generated name (Gather_70871, ReadValue_71348: the KV-cache plumbing in TinyLlama) say
      nothing about their module, so they are placed by their connections (option `absorb`, on by default):
      a node joins a module when ALL its consumers are in that module, else when ALL its producers are.
      This repeats until nothing changes, so a whole chain ReadValue -> Gather -> layer joins that layer.
      Nodes that DO have a module name (embeddings, final norm) are never moved this way.
    - Edges inside a module disappear; edges between boxes are merged into one edge
      labelled with how many edges it stands for ("17 edges").
    - `expand` lists modules to leave unfolded, to look inside one layer:
          collapse_modules:expand=layers.0
    - The folded nodes (id, name, op type, attributes, constant rows) are kept in node.meta["members"],
      and every edge that is no longer drawn as itself in node.meta["folded_edges"], so the node map JSON
      still lists them. Run collapse_fanout AFTER this pass (the other order also works, but the rows are clearer).
    """
    name = "collapse_modules"

    def __init__(self, pattern: str = r"(layers\.\d+)", expand: list[str] | set[str] | None = None,
                 absorb: bool = True) -> None:
        self.absorb = absorb
        self.regex = re.compile(pattern)
        if self.regex.groups < 1:
            raise ValueError("collapse_modules: pattern needs one capture group, e.g. (layers\\.\\d+)")
        self.expand = set(expand or ())

    @classmethod
    def from_options(cls, options: dict[str, list[str]]) -> "CollapseModulesPass":
        """Options: pattern=<regex with one group>  expand=layers.0,layers.1  absorb=true|false"""
        unknown = set(options) - {"pattern", "expand", "absorb"}
        if unknown:
            raise ValueError(f"collapse_modules: unknown option(s) {', '.join(sorted(unknown))}; "
                             "use pattern, expand, absorb")
        kwargs = {}
        if "absorb" in options:
            word = options["absorb"][-1].lower()
            if word not in ("true", "false"):
                raise ValueError("collapse_modules: absorb must be true or false")
            kwargs["absorb"] = word == "true"
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

    def _absorb(self, graph: Graph, module_of: dict[int, str]) -> None:
        """Give generated-name nodes a module from their neighbours (see the class docstring). Edits module_of."""
        consumers: dict[int, set[int]] = defaultdict(set)
        producers: dict[int, set[int]] = defaultdict(set)
        for e in graph.edges:
            consumers[e.src_id].add(e.dst_id)
            producers[e.dst_id].add(e.src_id)
        loose = {i for i, n in graph.nodes.items() if i not in module_of and AUTO_NAME.match(n.name)}
        while True:
            joined: dict[int, str] = {}  # decided from the state at the start of the round, so order never matters
            for i in loose:
                for neighbours in (consumers[i], producers[i]):
                    modules = {module_of.get(x) for x in neighbours}
                    if len(modules) == 1 and None not in modules:  # every neighbour is in the same module
                        joined[i] = modules.pop()
                        break
            if not joined:
                return
            module_of.update(joined)
            loose -= joined.keys()

    def run(self, graph: Graph) -> Graph:
        """Add one Group node per module, rewire the edges to it, then remove the folded nodes."""
        module_of: dict[int, str] = {}  # node id -> its module, from the name
        for n in graph.nodes.values():
            module = self._module_of(n)
            if module is not None:
                module_of[n.id] = module
        if self.absorb:
            self._absorb(graph, module_of)

        members: dict[str, list[Node]] = {}  # module -> its nodes, in order of first appearance
        for n in graph.nodes.values():
            module = module_of.get(n.id)
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
