#!/usr/bin/env python3
"""Fix the metadata of our converted Qwen3.6 GGUF so Ollama can load it.

Three bugs from convert_hf_to_gguf.py treating Qwen3.6 as a plain transformer:
  1. head_count_kv written as scalar (4) — must be a per-layer array
     (0 for linear_attention/SSM layers, 4 for full_attention).
  2. block_count = 65 — there are only 64 blocks of tensors (0..63);
     the +1 was a phantom nextn prediction layer.
  3. nextn_predict_layers = 1 — no such layer tensors exist; remove it.

Reuses llama.cpp's own copy_with_new_metadata so tensor data + alignment
are preserved byte-for-byte.
"""
from __future__ import annotations
import json
import sys

import gguf
from gguf.scripts.gguf_new_metadata import copy_with_new_metadata, MetadataDetails, get_field_data

if len(sys.argv) != 4:
    print(f"Usage: {sys.argv[0]} <input.gguf> <merged_config.json> <output.gguf>")
    sys.exit(1)

IN_GGUF, CONFIG_JSON, OUT_GGUF = sys.argv[1], sys.argv[2], sys.argv[3]

with open(CONFIG_JSON) as f:
    cfg = json.load(f)

layer_types = cfg["layer_types"]
n_layers = len(layer_types)
base_kv = cfg.get("num_key_value_heads", 4)
head_count_kv = [base_kv if t == "full_attention" else 0 for t in layer_types]
print(f"config: {n_layers} layers, "
      f"{sum(1 for t in layer_types if t == 'full_attention')} full_attention, "
      f"{sum(1 for t in layer_types if t == 'linear_attention')} linear_attention")
print(f"head_count_kv array (first 8): {head_count_kv[:8]} ... total {len(head_count_kv)}")

print(f"Loading: {IN_GGUF}")
reader = gguf.GGUFReader(IN_GGUF, "r")
arch = get_field_data(reader, gguf.Keys.General.ARCHITECTURE)
print(f"Architecture: {arch}")

writer = gguf.GGUFWriter(OUT_GGUF, arch=arch, endianess=reader.endianess)

alignment = get_field_data(reader, gguf.Keys.General.ALIGNMENT)
if alignment is not None:
    print(f"Custom alignment: {alignment}")
    writer.data_alignment = alignment

new_metadata = {
    f"{arch}.block_count": MetadataDetails(gguf.GGUFValueType.UINT32, n_layers),
    f"{arch}.attention.head_count_kv": MetadataDetails(
        gguf.GGUFValueType.ARRAY, head_count_kv, sub_type=gguf.GGUFValueType.INT32
    ),
}
remove_metadata = [f"{arch}.nextn_predict_layers"]

print(f"Setting block_count={n_layers}, head_count_kv=<array[{len(head_count_kv)}]>, "
      f"removing {remove_metadata[0]}")

copy_with_new_metadata(reader, writer, new_metadata, remove_metadata)
print(f"Wrote: {OUT_GGUF}")
