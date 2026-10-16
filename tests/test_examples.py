import sys
from pathlib import Path

import pytest

pytest.importorskip("openvino")
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

import make_models  # noqa: E402
import openvino as ov  # noqa: E402

from ovgraph.loaders import load_graph  # noqa: E402
from ovgraph.passes.hide_constants import HideConstantsPass  # noqa: E402


@pytest.mark.parametrize("build", [make_models.mini_cnn, make_models.mini_transformer])
def test_example_models_load_and_hide_constants(build, tmp_path):
    path = tmp_path / "m.xml"
    ov.save_model(build(), str(path), compress_to_fp16=False)
    g = load_graph(path)
    out = HideConstantsPass().run(g.clone())
    assert not any(n.op_type == "Constant" for n in out.nodes.values())
    assert len(out) < len(g)


def test_shared_weight_is_marked_in_transformer(tmp_path):
    path = tmp_path / "t.xml"
    ov.save_model(make_models.mini_transformer(), str(path), compress_to_fp16=False)
    out = HideConstantsPass().run(load_graph(path))
    rows = [t for n in out.nodes.values() for _, t in n.meta.get("extra_rows", [])]
    assert sum("(shared:" in r for r in rows) == 2  # shared_proj_w feeds v and out_proj


def test_fp16_export_gets_cleaner_with_fold_pass(tmp_path):
    from ovgraph.passes import Pipeline
    from ovgraph.passes.fold_constant_converts import FoldConstantConvertsPass

    path = tmp_path / "t16.xml"
    ov.save_model(make_models.mini_transformer(), str(path), compress_to_fp16=True)
    g = load_graph(path)
    plain = Pipeline([HideConstantsPass()]).run(g)
    folded = Pipeline([FoldConstantConvertsPass(), HideConstantsPass()]).run(g)
    count = lambda gr: sum(n.op_type == "Convert" for n in gr.nodes.values())  # noqa: E731
    assert count(plain) > 0 and count(folded) == 0
