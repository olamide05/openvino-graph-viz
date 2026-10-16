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


def test_cli_passes_follow_command_line_order_and_options(ir_model, tmp_path):
    from ovgraph.cli import build_parser, build_pipeline
    a = build_parser().parse_args([str(ir_model), "--highlight-name", "mm", "--hide-constants"])
    b = build_parser().parse_args([str(ir_model), "--hide-constants", "--pass", "highlight:names=mm;color=blue"])
    assert [p.name for p in build_pipeline(a).passes] == ["highlight", "hide_constants"]
    assert [p.name for p in build_pipeline(b).passes] == ["hide_constants", "highlight"]
    assert build_pipeline(b).passes[1].color == "blue"
    out = tmp_path / "o.svg"
    assert main([str(ir_model), "-o", str(out), "--hide-constants", "--highlight-name", "mm"]) == 0
    assert 'stroke="red"' in out.read_text(encoding="utf-8")


def test_cli_bad_pass_arguments_are_clean_errors(ir_model, tmp_path):
    import pytest
    assert main([str(ir_model), "--pass", "no-such-pass"]) == 1
    assert main([str(ir_model), "--pass", "highlight:bogus=1"]) == 1
    assert main([str(ir_model), "--highlight-id", "abc"]) == 1
    with pytest.raises(SystemExit):                      # malformed spec: argparse usage error
        main([str(ir_model), "--pass", "highlight:novalue"])


def test_corrupt_model_file_is_a_clean_error(tmp_path):
    bad = tmp_path / "broken.xml"
    bad.write_text("<this is not an openvino model>")
    assert main([str(bad)]) == 1
