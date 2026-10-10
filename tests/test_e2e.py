import shutil

import pytest

pytest.importorskip("openvino")
import numpy as np  # noqa: E402
import openvino as ov  # noqa: E402
import openvino.opset13 as ops  # noqa: E402

from ovgraph.cli import main  # noqa: E402
from ovgraph.loaders import load_graph  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("dot") is None, reason="Graphviz 'dot' not installed")


@pytest.fixture
def ir_model(tmp_path):
    x = ops.parameter([1, "?", 8], np.float32, name="x")
    w = ops.constant(np.ones((8, 8), np.float32), name="w")
    mm = ops.matmul(x, w, False, False, name="mm")
    add = ops.add(mm, w, name="add")  # w is shared by MatMul and Add
    model = ov.Model([ops.result(add, name="out")], [x], "tiny")
    path = tmp_path / "tiny.xml"
    ov.save_model(model, str(path), compress_to_fp16=False)
    return path


def test_loader_reads_ports_shapes_and_dynamic_dims(ir_model):
    g = load_graph(ir_model)
    by_name = {n.name: n for n in g.nodes.values()}
    mm = by_name["mm"].id
    ins = {e.dst_port: e for e in g.in_edges(mm)}
    assert ins[0].shape == (1, None, 8) and ins[0].dtype == "f32"
    assert ins[1].shape == (8, 8)
    assert {n.op_type for n in g.nodes.values()} >= {"Parameter", "Constant", "MatMul", "Add", "Result"}


def test_cli_end_to_end_hide_constants_and_highlight(ir_model, tmp_path):
    out = tmp_path / "out.svg"
    rc = main([str(ir_model), "-o", str(out), "--hide-constants", "--highlight-op", "MatMul"])
    assert rc == 0
    svg = out.read_text(encoding="utf-8")
    assert "<svg" in svg
    assert "(shared:" in svg           # w feeds MatMul and Add
    assert "f32[8,8]" in svg           # shape kept
    assert "stroke=\"red\"" in svg     # highlighted node
    assert "Constant" not in svg       # constant node itself is hidden


def test_cli_missing_file_is_a_clean_error(tmp_path):
    assert main([str(tmp_path / "nope.xml")]) == 1


def test_cli_unknown_extension_is_a_clean_error(tmp_path):
    f = tmp_path / "m.onnx"
    f.write_text("x")
    assert main([str(f)]) == 1
