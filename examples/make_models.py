"""Build two small example OpenVINO IR models (no downloads needed).

Run:  python examples/make_models.py
Writes examples/models/mini_cnn.xml/.bin and examples/models/mini_transformer.xml/.bin
"""
from pathlib import Path

import numpy as np
import openvino as ov
import openvino.opset13 as ops

OUT = Path(__file__).parent / "models"
rng = np.random.default_rng(0)


def const(*shape, name):
    return ops.constant(rng.standard_normal(shape).astype(np.float32) * 0.1, name=name)


def mini_cnn() -> ov.Model:
    """Conv -> ReLU -> MaxPool -> Conv -> ReLU -> Flatten -> MatMul -> Softmax."""
    x = ops.parameter([1, 3, 32, 32], np.float32, name="image")
    h = ops.convolution(x, const(8, 3, 3, 3, name="conv1_w"), [1, 1], [1, 1], [1, 1], [1, 1], name="conv1")
    h = ops.add(h, const(1, 8, 1, 1, name="conv1_b"), name="conv1_bias")
    h = ops.relu(h, name="relu1")
    h = ops.max_pool(
        h, strides=[2, 2], dilations=[1, 1], pads_begin=[0, 0], pads_end=[0, 0], kernel_shape=[2, 2], name="pool1"
    ).output(0)
    h = ops.convolution(h, const(16, 8, 3, 3, name="conv2_w"), [1, 1], [1, 1], [1, 1], [1, 1], name="conv2")
    h = ops.add(h, const(1, 16, 1, 1, name="conv2_b"), name="conv2_bias")
    h = ops.relu(h, name="relu2")
    h = ops.reshape(h, ops.constant(np.array([1, -1], np.int64), name="flat_shape"), False, name="flatten")
    h = ops.matmul(h, const(16 * 16 * 16, 10, name="fc_w"), False, False, name="fc")
    h = ops.add(h, const(1, 10, name="fc_b"), name="fc_bias")
    h = ops.softmax(h, 1, name="probs")
    return ov.Model([ops.result(h, name="out")], [x], "mini_cnn")


def mini_transformer(d=32, heads=2, seq="?") -> ov.Model:
    """One attention block (dynamic sequence length) + MLP, with residuals and a shared weight."""
    x = ops.parameter([1, seq, d], np.float32, name="tokens")
    shared = const(d, d, name="shared_proj_w")  # used twice -> shows the (shared: ...) marker

    def proj(inp, w, name):
        return ops.matmul(inp, w, False, False, name=name)

    q = proj(x, const(d, d, name="wq"), "q")
    k = proj(x, const(d, d, name="wk"), "k")
    v = proj(x, shared, "v")
    scores = ops.matmul(q, k, False, True, name="scores")
    scale = ops.constant(np.array(1.0 / np.sqrt(d), np.float32), name="scale")
    probs = ops.softmax(ops.multiply(scores, scale, name="scaled"), -1, name="attn_probs")
    ctx = ops.matmul(probs, v, False, False, name="context")
    out = proj(ctx, shared, "out_proj")
    res1 = ops.add(x, out, name="residual1")
    mu = ops.reduce_mean(res1, ops.constant(np.array([-1], np.int64), name="ln_axes"), True, name="ln_mean")
    norm = ops.subtract(res1, mu, name="ln_center")
    ff = ops.matmul(norm, const(d, 4 * d, name="ff1_w"), False, False, name="ff1")
    ff = ops.gelu(ff, "ERF", name="gelu")
    ff = ops.matmul(ff, const(4 * d, d, name="ff2_w"), False, False, name="ff2")
    res2 = ops.add(norm, ff, name="residual2")
    return ov.Model([ops.result(res2, name="out")], [x], "mini_transformer")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, model in {"mini_cnn": mini_cnn(), "mini_transformer": mini_transformer()}.items():
        ov.save_model(model, str(OUT / f"{name}.xml"), compress_to_fp16=False)
        print("saved", OUT / f"{name}.xml", f"({len(model.get_ordered_ops())} ops)")
