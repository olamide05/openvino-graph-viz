from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .loaders import available_loaders, load_graph
from .passes import HideConstantsPass, HighlightPass, Pipeline, available_passes, get_pass
from .render import GraphvizRenderer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ovgraph", description="Render a model graph as SVG.")
    p.add_argument("model", help="path to the model file (OpenVINO IR: the .xml)")
    p.add_argument("-o", "--output", help="output SVG path (default: <model>.svg)")
    p.add_argument("--format", choices=available_loaders(), help="force a loader (default: by extension)")
    p.add_argument("--hide-constants", action="store_true", help="hide Constant nodes, show them as rows")
    p.add_argument("--highlight-op", action="append", default=[], metavar="OP",
                   help="red border on every node of this op type (repeatable)")
    p.add_argument("--highlight-id", action="append", default=[], type=int, metavar="ID",
                   help="red border on the node with this id (repeatable)")
    p.add_argument("--pass", dest="extra_passes", action="append", default=[], metavar="NAME",
                   help=f"run another registered pass by name ({', '.join(available_passes())})")
    p.add_argument("--attrs", action="store_true", help="show node attributes")
    p.add_argument("--no-edge-labels", action="store_true", help="omit dtype/shape on edges (faster, smaller)")
    p.add_argument("--timing", action="store_true", help="print load/transform/render timings")
    return p


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    pipe = Pipeline()
    if args.hide_constants:
        pipe.add(HideConstantsPass())
    for name in args.extra_passes:
        if name in ("hide_constants", "highlight"):
            continue  # have dedicated flags
        pipe.add(get_pass(name)())
    if args.highlight_op or args.highlight_id:
        pipe.add(HighlightPass(node_ids=set(args.highlight_id), op_types=set(args.highlight_op)))
    return pipe


def run(args: argparse.Namespace) -> dict[str, float]:
    t0 = time.perf_counter()
    graph = load_graph(args.model, args.format)
    t1 = time.perf_counter()
    out = build_pipeline(args).run(graph)
    t2 = time.perf_counter()
    path = Path(args.output) if args.output else Path(args.model).with_suffix(".svg")
    GraphvizRenderer(edge_labels=not args.no_edge_labels, show_attributes=args.attrs).render_svg(out, path)
    t3 = time.perf_counter()
    return {"nodes_in": len(graph), "nodes_out": len(out), "edges_out": len(out.edges),
            "load_s": t1 - t0, "transform_s": t2 - t1, "render_s": t3 - t2, "output": str(path)}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        info = run(args)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"wrote {info['output']} ({info['nodes_in']} nodes loaded, {info['nodes_out']} drawn)")
    if args.timing:
        print(f"load {info['load_s']:.3f}s | transform {info['transform_s']:.3f}s | render {info['render_s']:.3f}s",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
