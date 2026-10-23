from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from .loaders import available_loaders, load_graph
from .passes import Pipeline, available_passes, get_pass
from .export import write_node_map
from .render import GraphvizRenderer


def parse_pass_spec(spec: str) -> tuple[str, dict[str, list[str]]]:
    """'highlight:names=a,b;color=blue' -> ('highlight', {'names': ['a', 'b'], 'color': ['blue']}).

    The pass name ends at the first ':'; options are 'key=v1,v2' separated by ';'.
    (Node names that contain ',' or ';' should use --highlight-name instead.)
    """
    name, _, rest = spec.partition(":")
    name = name.strip()
    if not name:
        raise ValueError(f"empty pass name in {spec!r}")
    options: dict[str, list[str]] = {}
    for part in filter(None, (p.strip() for p in rest.split(";"))):
        key, eq, value = part.partition("=")
        if not eq or not key.strip():
            raise ValueError(f"bad option {part!r} in {spec!r}; expected key=value")
        options.setdefault(key.strip(), []).extend(v for v in value.split(",") if v != "")
    return name, options


def _step(pass_name: str | None, key: str | None = None, flag: bool = False):
    """argparse action that appends a pass to args.steps IN COMMAND-LINE ORDER.

    All the pass flags share args.steps, so `--hide-constants --highlight-op MatMul`
    runs hide first and `--highlight-op MatMul --hide-constants` runs highlight first.
    """
    class Step(argparse.Action):
        def __init__(self, option_strings, dest, **kw):
            super().__init__(option_strings, dest, nargs=0 if flag else None, **kw)

        def __call__(self, parser, ns, values, option_string=None):
            steps = list(getattr(ns, "steps", None) or [])
            if pass_name is None:  # --pass SPEC
                try:
                    steps.append(parse_pass_spec(values))
                except ValueError as e:  # argparse turns this into a clean usage error (exit code 2)
                    raise argparse.ArgumentError(self, str(e)) from None
            else:
                steps.append((pass_name, {} if flag or key is None else {key: [values]}))
            ns.steps = steps
    return Step


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ovgraph", description="Render a model graph as SVG.",
        epilog="Passes run in the order their flags appear on the command line.")
    p.add_argument("model", help="path to the model file (OpenVINO IR: the .xml)")
    p.add_argument("-o", "--output", help="output SVG path (default: <model>.svg)")
    p.add_argument("--format", choices=available_loaders(), help="force a loader (default: by extension)")
    p.add_argument("--hide-constants", action=_step("hide_constants", flag=True), dest="steps", default=[],
                   help="hide Constant nodes, show them as rows")
    p.add_argument("--collapse-fanout", action=_step("collapse_fanout", flag=True), dest="steps", default=[],
                   help="stop drawing edges of nodes with more than 3 consumers (use --pass collapse_fanout:threshold=N "
                        "for another limit); run after --hide-constants")
    p.add_argument("--collapse-modules", action=_step("collapse_modules", flag=True), dest="steps", default=[],
                   help="one box per transformer layer (name pattern layers.N); see --pass collapse_modules:expand=layers.0")
    p.add_argument("--highlight-name", action=_step("highlight", "names"), dest="steps", default=[],
                   metavar="NAME", help="red border on every node with this name (repeatable)")
    p.add_argument("--highlight-op", action=_step("highlight", "ops"), dest="steps", default=[],
                   metavar="OP", help="red border on every node of this op type (repeatable)")
    p.add_argument("--highlight-id", action=_step("highlight", "ids"), dest="steps", default=[],
                   metavar="ID", help="red border on the node with this id (repeatable)")
    p.add_argument("--pass", action=_step(None), dest="steps", default=[], metavar="SPEC",
                   help="run any registered pass: NAME or NAME:key=value;key=value "
                        f"(available: {', '.join(available_passes())}). Repeatable.")
    p.add_argument("--attrs", action="store_true", help="show node attributes")
    p.add_argument("--hide-names", action="store_true",
                   help="leave node names out of the picture (narrower boxes); pair with --node-map")
    p.add_argument("--node-map", metavar="JSON",
                   help="write a JSON file: node id -> name, op type, attributes, rows, collapsed edges")
    p.add_argument("--no-edge-labels", action="store_true", help="omit dtype/shape on edges (faster, smaller)")
    p.add_argument("--fast", action=argparse.BooleanOptionalAction, default=None,
                   help="fast layout (straight edges). Default: automatic for big graphs")
    p.add_argument("--timing", action="store_true",
                   help="print graph size, load/transform/render timings and the environment")
    return p


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    """One pass per step, in the order the user gave them. Unknown passes/options raise."""
    pipe = Pipeline()
    for name, options in (args.steps or []):
        pipe.add(get_pass(name).from_options(options))
    return pipe


def environment_summary() -> str:
    """One line describing the machine, for the README performance table."""
    parts = [platform.platform(), f"Python {platform.python_version()}", f"{os.cpu_count()} logical CPUs"]
    try:
        import openvino
        parts.append(f"openvino {openvino.__version__.split('-')[0]}")
    except ImportError:
        pass
    try:
        v = subprocess.run(["dot", "-V"], capture_output=True, text=True, timeout=10)
        parts.append((v.stderr or v.stdout).strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return "; ".join(parts)


def run(args: argparse.Namespace) -> dict[str, float]:
    t0 = time.perf_counter()
    graph = load_graph(args.model, args.format)
    t1 = time.perf_counter()
    out = build_pipeline(args).run(graph)
    t2 = time.perf_counter()
    path = Path(args.output) if args.output else Path(args.model).with_suffix(".svg")
    if args.node_map:
        write_node_map(out, args.node_map)
    GraphvizRenderer(edge_labels=not args.no_edge_labels, show_attributes=args.attrs, fast=args.fast,
                     show_names=not args.hide_names).render_svg(out, path)
    t3 = time.perf_counter()
    return {"nodes_in": len(graph), "edges_in": len(graph.edges),
            "nodes_out": len(out), "edges_out": len(out.edges),
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
        print(f"graph: {info['nodes_in']} nodes, {info['edges_in']} edges -> "
              f"drawn: {info['nodes_out']} nodes, {info['edges_out']} edges", file=sys.stderr)
        print(f"load {info['load_s']:.3f}s | transform {info['transform_s']:.3f}s | render {info['render_s']:.3f}s",
              file=sys.stderr)
        print(f"environment: {environment_summary()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
