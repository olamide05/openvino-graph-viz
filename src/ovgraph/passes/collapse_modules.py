"""Pass that folds all nodes of one module (for example one transformer layer) into a single box.
"""
from __future__ import annotations

import re
from collections import defaultdict

from ..graph import Edge, Graph, Node
from .base import Pass, last_value, register_pass

GROUP = "Group"
SHARED = "shared (masks, positions)"
# OpenVINO's generated names look like Gather_70871 (op type, underscore, number). Such a name says
# nothing about which module the node belongs to, unlike __module.model.layers.7.self_attn/...
AUTO_NAME = re.compile(r"^[A-Za-z0-9]+_\d+$")
# The model's inputs and outputs are never moved into a box by absorb or group_shared, so they stay visible
# (TinyLlama's logits output is called Result_24245, a generated name).
KEEP_OUTSIDE = {"Parameter", "Result"}


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
      Nodes that DO have a module name (embeddings, final norm) are never moved this way, and neither
      are Parameter and Result nodes (the model's inputs and outputs stay visible).
    - Generated-name nodes that are still loose after that (shared by several layers: attention mask and
      position id plumbing) go into one box "shared (masks, positions)" (option `group_shared`, on by default;
      it only applies together with `absorb`, and only if the pattern matched at least one node, so a model
      without layers.N names comes out unchanged instead of as one big box).
      The shared box can be upstream of one layer and downstream of another. On TinyLlama this makes no
      cycle (checked), but in a model where a generated-name node both reads a layer's output and feeds
      the layers, the boxes form a cycle (layers.0 -> shared -> layers.0); Graphviz still draws it.
    - Edges inside a module disappear. Edges into or out of a box are merged per pair of ends into one edge
      labelled with how many edges it stands for ("17 edges"). A box has one input and one output port
      (port 0), but the end on a real node keeps its real port. If only ONE edge is merged it is drawn like
      a normal edge, with its dtype and shape instead of "1 edge".
    - `expand` lists modules to leave unfolded, to look inside one layer:
          collapse_modules:expand=layers.0
    - The folded nodes (id, name, op type, attributes, constant rows) are kept in node.meta["members"],
      and every edge that is no longer drawn as itself in node.meta["folded_edges"], so the node map JSON
      still lists them. A member's border_color (from highlight, run before this pass) goes to its box.
    - Run collapse_fanout AFTER this pass. The other order loses nothing, but then the wide nodes' edges are
      removed before the boxes are built, so the boxes show neither those edges nor rows for them: that
      connectivity is only in the node map JSON (under "collapsed_outputs" of the box).
    """
    name = "collapse_modules"

    def __init__(self, pattern: str = r"(layers\.\d+)", expand: list[str] | set[str] | None = None,
                 absorb: bool = True, group_shared: bool = True) -> None:
        self.absorb = absorb
        self.group_shared = group_shared
        self.regex = re.compile(pattern)
        if self.regex.groups < 1:
            raise ValueError("collapse_modules: pattern needs one capture group, e.g. (layers\\.\\d+)")
        self.expand = set(expand or ())

    @classmethod
    def from_options(cls, options: dict[str, list[str]]) -> "CollapseModulesPass":
        """Options: pattern=<regex with one group>  expand=layers.0,layers.1  absorb=true|false"""
        unknown = set(options) - {"pattern", "expand", "absorb", "group_shared"}
        if unknown:
            raise ValueError(f"collapse_modules: unknown option(s) {', '.join(sorted(unknown))}; "
                             "use pattern, expand, absorb, group_shared")
        kwargs = {}
        if "absorb" in options:
            word = last_value("collapse_modules", options, "absorb").lower()
            if word not in ("true", "false"):
                raise ValueError("collapse_modules: absorb must be true or false")
            kwargs["absorb"] = word == "true"
        if "group_shared" in options:
            word = last_value("collapse_modules", options, "group_shared").lower()
            if word not in ("true", "false"):
                raise ValueError("collapse_modules: group_shared must be true or false")
            kwargs["group_shared"] = word == "true"
        if "pattern" in options:
            last_value("collapse_modules", options, "pattern")  # only to reject an empty pattern= clearly
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
        loose = {i for i, n in graph.nodes.items()
                 if i not in module_of and AUTO_NAME.match(n.name) and n.op_type not in KEEP_OUTSIDE}
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
        # Without any module there is nothing to absorb into and nothing is "shared between layers".
        if self.absorb and module_of:
            self._absorb(graph, module_of)
            if self.group_shared:  # what is still loose is shared between layers (masks, position ids)
                for i, n in graph.nodes.items():
                    if i not in module_of and AUTO_NAME.match(n.name) and n.op_type not in KEEP_OUTSIDE:
                        module_of[i] = SHARED

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
            colors = [n.meta["border_color"] for n in nodes if "border_color" in n.meta]
            if colors:  # a highlighted member highlights its box (the first member's colour wins)
                group.meta["border_color"] = colors[0]
            graph.add_node(group)
            for n in nodes:
                group_of[n.id] = group.id
            next_id += 1

        kept: list[Edge] = []
        # (source, source port, destination, destination port) -> the edges merged into one drawn edge.
        # A box end always uses port 0; a real node keeps its real port.
        merged: dict[tuple[int, int, int, int], list[Edge]] = defaultdict(list)
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
                src_port = 0 if e.src_id in group_of else e.src_port
                dst_port = 0 if e.dst_id in group_of else e.dst_port
                merged[(src, src_port, dst, dst_port)].append(e)
            # else: both ends are inside the same module; the edge is hidden
        for (src, src_port, dst, dst_port), edges in merged.items():
            if len(edges) == 1:  # just one edge: draw it as itself, with its dtype and shape
                kept.append(Edge(src, src_port, dst, dst_port, edges[0].shape, edges[0].dtype))
            else:
                kept.append(Edge(src, src_port, dst, dst_port, label=f"{len(edges)} edges"))

        for node_id in group_of:
            del graph.nodes[node_id]
        graph.edges = kept
        return graph
