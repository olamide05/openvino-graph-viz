# ovgraph: OpenVINO model graph visualizer

Loads an OpenVINO IR model (`.xml` + `.bin`), turns it into a small framework-independent graph,
optionally transforms it with configurable passes, and draws it as an SVG with Graphviz.

![before and after](examples/svg/fp16_before.svg)

* Original model is never modified (every pass runs on a copy).
* Built-in passes: **highlight** (red border) and **hide constants** (constants become rows on the node that uses them).
* Extra example pass: **fold_constant_converts**, which cleans up FP16 exports such as TinyLlama.

## 1. Getting started

Requirements: Python 3.10+, and the [Graphviz](https://graphviz.org/download/) `dot` program on your PATH
(check with `dot -V`).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate
pip install -e ".[openvino,dev]"
python -m pytest -q                 # 32 tests
```

Try it on the bundled example models:

```powershell
python -m ovgraph examples\models\mini_transformer.xml -o out.svg --hide-constants --highlight-op MatMul
```

## 2. Usage

```
ovgraph MODEL.xml [-o OUT.svg] [--hide-constants] [--highlight-op OP]... [--highlight-id ID]...
                  [--pass NAME]... [--attrs] [--no-edge-labels] [--fast | --no-fast] [--timing]
```

| Flag | What it does |
|---|---|
| `--hide-constants` | Remove Constant nodes. Each consumer gets a row per constant input: `[1]: f32[768,768]`. If a constant feeds several consumers the row ends with `(shared: <node id>)`. Only dtype and shape are shown, never values. |
| `--highlight-op OP` | Red border on every node of op type `OP` (repeatable, e.g. `MatMul`). |
| `--highlight-id ID` | Red border on the node with this id (the number shown in each box, e.g. `MatMul (id 7)`). |
| `--pass NAME` | Run another registered pass, e.g. `fold_constant_converts`. These run before `--hide-constants`. |
| `--attrs` | Show node attributes (hidden by default, they are long). |
| `--no-edge-labels` | Leave dtype/shape off the edges (smaller and faster). |
| `--fast` / `--no-fast` | Force the fast layout on or off. Default: automatic above 400 drawn nodes. |
| `--timing` | Print load / transform / render time to stderr. |

Each box shows the node name, `OpType (id N)`, input ports on top, output ports at the bottom, and any
rows added by passes. Edges carry `dtype[shape]`; a `?` is a dynamic dimension.

As a library:

```python
from ovgraph.loaders import load_graph
from ovgraph.passes import Pipeline, HideConstantsPass, HighlightPass
from ovgraph.render import GraphvizRenderer

g = load_graph("model.xml")
g2 = Pipeline([HideConstantsPass(), HighlightPass(op_types={"MatMul"})]).run(g)   # g is untouched
GraphvizRenderer().render_svg(g2, "out.svg")
```

## 3. Architecture

```
model file --> Loader --> Graph (Nodes + Edges) --> Pipeline of Passes --> Renderer --> SVG
```

| Module | Job |
|---|---|
| `graph.py` | `Node` (int id, name, op type, attributes, `meta` for notes between passes and renderer), `Edge` (src/dst id and **port**, shape, dtype; frozen), `Graph`. |
| `loaders/` | Turn a file into a `Graph`. Only `openvino_loader.py` imports OpenVINO (lazily). |
| `passes/` | `Pass` base class, `Pipeline`, and a name registry. |
| `render/graphviz_renderer.py` | Graph to Graphviz HTML-like table labels with a port cell per input/output, so edges attach to the right port. |
| `cli.py` | Argument parsing and wiring. |

Design choices:

* **Graph is independent of OpenVINO.** Only the loader knows OpenVINO, so passes and the renderer are testable without a model.
* **Nodes are keyed by an integer id, not by name**, because names are not unique.
* **Edges are a list, not a dict keyed by `(src, dst)`**, because two nodes can be joined by several edges (different ports).
* **Passes never mutate the input.** `Pipeline.run` clones the graph first (deep copy of nodes, shallow copy of the edge list because `Edge` is frozen).
* **Passes talk to the renderer through `node.meta`** (`border_color`, `extra_rows`), so the renderer does not need to know about specific passes.

## 4. Extensibility

Add a pass: subclass `Pass`, implement `run`, decorate it. It is then available as `--pass NAME`.

```python
from ovgraph.passes import Pass, register_pass

@register_pass
class MyPass(Pass):
    name = "my_pass"
    def run(self, graph):          # graph is already a private copy; mutate and return it
        for n in graph.nodes.values():
            if n.op_type == "Softmax":
                n.meta["border_color"] = "blue"
        return graph
```

Add a model format: subclass `Loader` (`name`, `extensions`, `load(path) -> Graph`) and decorate it with `@register_loader`;
it is picked by file extension or `--format`. Nothing else changes.

### The custom pass: `fold_constant_converts`

FP16 exports (like TinyLlama) store each weight as `Constant(f16) -> Convert(f32) -> layer`.
Hiding constants alone leaves one useless Convert box per weight. This pass removes a Convert whose only input is a
Constant and joins the Constant directly to the Convert's consumers with the Convert's output dtype, so `hide_constants`
then shows `[1]: f32[2048,2048]`. Converts on real activations are left alone.

```powershell
python -m ovgraph models\tinyllama\openvino_model.xml -o tinyllama_before.svg --hide-constants
python -m ovgraph models\tinyllama\openvino_model.xml -o tinyllama_after.svg  --hide-constants --pass fold_constant_converts
```

On TinyLlama this takes the drawn graph from **1911 to 1528 nodes**. Small demo:
`examples/svg/fp16_before.svg` vs `examples/svg/fp16_after.svg`.

## 5. Examples

Three models, with generated SVGs:

| Model | How to get it | SVGs |
|---|---|---|
| `mini_cnn` | `python examples/make_models.py` | `examples/svg/mini_cnn_full.svg`, `mini_cnn_hidden.svg` |
| `mini_transformer` (dynamic sequence length, one shared weight) | same script | `examples/svg/mini_transformer_full.svg`, `mini_transformer_hidden.svg` |
| TinyLlama-1.1B-Chat (3168 nodes) | `optimum-cli export openvino --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --weight-format fp16 models/tinyllama` | `tinyllama_before.svg`, `tinyllama_after.svg` (hidden constants, without / with the custom pass) |

Regenerate the small SVGs with `python examples/generate_svgs.py`.

## 6. Testing

```powershell
python -m pytest -q      # 32 tests, about 6 s
```

* Unit tests: graph invariants, each pass (including that the input graph is untouched), renderer output.
* End-to-end tests build a small IR with OpenVINO, run the real CLI and check the SVG (needs `dot`; skipped without it).
* Error paths: missing file and unknown extension give a clean message and exit code 1.
* The example models and the custom pass are covered too.

## 7. Performance

Measured on the development laptop (Windows, Python 3.14, Graphviz 16.1).

Synthetic chains (`python examples/benchmark.py`; about 40% constants), hide + highlight passes, fast render:

| nodes | edges | passes | drawn | render |
|---|---|---|---|---|
| 501 | 665 | 0.002 s | 334 | 0.81 s |
| 999 | 1329 | 0.006 s | 666 | 0.95 s |
| 2499 | 3329 | 0.015 s | 1666 | 0.98 s |
| 5001 | 6665 | 0.031 s | 3334 | 1.55 s |

TinyLlama (3168 nodes, `--hide-constants`): load 0.47 s, transform 0.02 s, render 9.6 s.

* Loading and passes are cheap; **Graphviz layout is the bottleneck**.
* Curved edge routing was the main cost. Above 400 drawn nodes the renderer switches to a fast layout
  (`splines=line`, reduced `nslimit`/`mclimit`, no `remincross`). Without it, the full TinyLlama graph did not finish in a reasonable time.
* The benchmark caught an O(E^2) loop in `hide_constants` (it looked up each shared constant's consumers once per edge).
  It now counts consumers in one pass, which made that pass about 20x faster at 5k nodes.

## 8. Known limitations

* Only OpenVINO IR (`.xml`) is supported; other formats need a new loader.
* Subgraphs (`Loop`, `TensorIterator`, `If` bodies) are not expanded: they appear as a single node.
* Fast layout uses straight edges, so edges can cross nodes on big graphs. `--no-fast` gives nicer routing but is far slower.
* Constant values are never read; only dtype and shape are shown.
* `fold_constant_converts` assumes the `Constant -> Convert` pattern; other quantisation patterns (e.g. decompression with Multiply/Subtract) are not folded.
* Finding the in/out edges of one node scans the whole edge list (O(E)). Fine at this size; an adjacency index is the next step if profiling shows a need.
* Very large SVGs (several MB for TinyLlama) are slow to open in some viewers; use a browser.
