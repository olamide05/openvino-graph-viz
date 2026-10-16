"""Render before/after SVGs for the small example models.

Run (from the repo root, after make_models.py):  python examples/generate_svgs.py
"""
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from ovgraph.cli import main

HERE = Path(__file__).parent
MODELS = HERE / "models"
OUT = HERE / "svg"
OUT.mkdir(exist_ok=True)

JOBS = {
    "mini_cnn": ["--highlight-op", "Convolution"],
    "mini_transformer": ["--highlight-op", "MatMul"],
}

for name, extra in JOBS.items():
    model = str(MODELS / f"{name}.xml")
    main([model, "-o", str(OUT / f"{name}_full.svg")])
    main([model, "-o", str(OUT / f"{name}_hidden.svg"), "--hide-constants", *extra])

# Custom pass demo: an FP16 export has Constant -> Convert chains. Compare without / with the pass.
import openvino as ov  # noqa: E402

import make_models  # noqa: E402

fp16 = str(MODELS / "mini_transformer_fp16.xml")
ov.save_model(make_models.mini_transformer(), fp16, compress_to_fp16=True)
main([fp16, "-o", str(OUT / "fp16_before.svg"), "--hide-constants"])
main([fp16, "-o", str(OUT / "fp16_after.svg"), "--hide-constants", "--pass", "fold_constant_converts",
      "--highlight-op", "MatMul"])
