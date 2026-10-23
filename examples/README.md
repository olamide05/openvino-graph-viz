# Examples

Generated from the TinyLlama OpenVINO IR with the real OpenVINO loader (see README section 4b for the exact commands):

* `tinyllama_overview.svg` / `.json`: one box per layer plus the embedding, rotary embedding and mask building, names hidden, fan-out collapsed. The JSON maps every node id to its name, attributes and edges.
* `tinyllama_layer0.svg`: the same, with `layers.0` left open to look inside one layer.
