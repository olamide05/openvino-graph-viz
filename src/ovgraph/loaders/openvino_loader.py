"""Loader for OpenVINO IR models (.xml + .bin). The only module that imports OpenVINO.
"""
from __future__ import annotations

from pathlib import Path

from ..graph import Edge, Graph, Node
from .base import Loader, register_loader

MAX_ATTR_LEN = 80


def _shape(partial_shape):
    """OpenVINO PartialShape -> tuple of int|None, or None if the rank is unknown."""
    if not partial_shape.rank.is_static:
        return None
    return tuple(d.get_length() if d.is_static else None for d in partial_shape)


def _attributes(op) -> dict[str, str]:
    # Never ask a Constant for its attributes: that can materialise the tensor.
    if op.get_type_name() == "Constant":
        return {}
    try:
        raw = op.get_attributes()
    except Exception:  # some ops cannot report attributes; that is not fatal
        return {}
    out = {}
    for k, v in raw.items():
        s = str(v)
        out[str(k)] = s if len(s) <= MAX_ATTR_LEN else s[: MAX_ATTR_LEN - 3] + "..."
    return out


@register_loader
class OpenVINOLoader(Loader):
    """Reads an OpenVINO IR model and builds a Graph: one Node per op, one Edge per op input."""

    name = "openvino"
    extensions = (".xml",)  # OpenVINO IR = .xml (graph) + .bin (weights)

    def load(self, path: Path) -> Graph:
        """Missing file raises FileNotFoundError; an unreadable model raises ValueError."""
        import openvino as ov  # lazy: only needed when this loader is actually used

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"model file not found: {path}")
        try:
            model = ov.Core().read_model(str(path))
        except RuntimeError as e:  # corrupt or unsupported file; keep it a clean error
            raise ValueError(f"cannot read OpenVINO model {path}: {str(e).splitlines()[0]}") from e
        graph = Graph()
        ops = model.get_ordered_ops()

        # Our id for each op. get_instance_id() is stable per op inside a model,
        # whereas names can be duplicated or missing.
        ids = {op.get_instance_id(): i for i, op in enumerate(ops)}

        for i, op in enumerate(ops):
            graph.add_node(Node(i, op.get_friendly_name(), op.get_type_name(), _attributes(op)))

        for dst_id, op in enumerate(ops):
            for inp in op.inputs():
                src = inp.get_source_output()
                graph.add_edge(Edge(
                    src_id=ids[src.get_node().get_instance_id()],
                    src_port=src.get_index(),
                    dst_id=dst_id,
                    dst_port=inp.get_index(),
                    shape=_shape(src.get_partial_shape()),
                    dtype=src.get_element_type().get_type_name(),
                ))
        return graph
