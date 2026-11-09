#!/usr/bin/env python3
"""Write the configuration file that lets official vLLM load GLM-5.3-Flash NVFP4's MTP layer.

    python3 official_mtp_config.py MODEL_DIR OUT.yaml

The checkpoint stores its MTP layer (the decoder layer after the last one) in BF16 without listing it as
unquantized, so official vLLM tries to load it as NVFP4 and fails. This adds three `ignore` entries for that layer to
the checkpoint's quantization config through `hf-overrides`; no file of the checkpoint changes. JSON is valid YAML.
"""
import json
import sys

model, out = sys.argv[1:3]
config = json.load(open(f"{model}/config.json"))
layer = (config.get("text_config") or config)["num_hidden_layers"]
quant = dict(config["quantization_config"])
quant["ignore"] = list(quant.get("ignore") or []) + [
    f"model.language_model.layers.{layer}*", f"model.layers.{layer}*", f"*.layers.{layer}.*"]
json.dump({"hf-overrides": {"quantization_config": quant}}, open(out, "w"))
