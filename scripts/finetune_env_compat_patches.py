"""Idempotent post-install patches for the fine-tune venv.

Pinned versions in `scripts/requirements-finetune.txt` (trl 0.24.0 +
transformers 5.3.0 + llm_blender 0.0.2) have two known incompatibilities
that the upstream maintainers haven't fixed and likely never will (trl
moves on; llm_blender hasn't shipped a new release since 2024).

This script applies the minimal source patches needed to make the stack
import cleanly. Run it after every `pip install -r requirements-finetune.txt`
(reinstalls overwrite the patched site-packages files).

Usage (from the project root, with the finetune venv on PATH or activated):
    python scripts/finetune_env_compat_patches.py

Behaviour: idempotent. Each patch checks for a marker string before applying
and prints "already patched" if it's already in place. No-ops if the venv
files don't exist.

Patch 1 — trl/import_utils.py
  transformers 5.x changed `_is_package_available()` to always return a
  (bool, version) tuple. trl 0.24.0 assigns the raw return value to
  bare `_<pkg>_available` globals and uses them as `if is_X_available():`,
  treating the always-truthy tuple as True. Result: trl tries to import
  weave / vllm / fastapi / liger_kernel even when uninstalled.
  Fix: wrap `_is_package_available` to unwrap the tuple when
  `return_version=False` (the common case).

Patch 2 — llm_blender/blender/{blender,blender_utils}.py
  llm_blender 0.0.2 does `from transformers.utils.hub import TRANSFORMERS_CACHE`,
  but transformers 5.x removed that constant (replaced by `HF_HUB_CACHE` in
  huggingface_hub.constants).
  Fix: try-import with fallback to the new name.

Patch 3 — transformers/models/qwen3_5/modeling_qwen3_5.py
  Qwen3.5 ships as a multimodal arch (`Qwen3_5ForConditionalGeneration`)
  even when only text weights are wired into the LoRA path. During text-only
  training the cached `rope_deltas` from initialization has the wrong shape,
  and compute_3d_position_ids crashes at `position_ids + delta` with a
  shape-mismatch RuntimeError. Add an early-exit at the top of the function:
  when no multimodal signals are present (no image_grid_thw, no
  video_grid_thw, no mm_token_type_ids), return None and let downstream
  use cache_position to derive 1D positions.
"""
from __future__ import annotations

import sys
import sysconfig
from pathlib import Path


_TRL_MARKER = "# Patch for transformers 5.x compat: _is_package_available now always returns"
_BLENDER_MARKER = "# transformers >= 5.0"
_QWEN_MARKER = "Patch 2026-05-13 (Nova task #44): text-only fast-exit"


def _site_packages() -> Path:
    """Resolve the active venv's site-packages directory."""
    return Path(sysconfig.get_paths()["purelib"])


def _patch_trl_import_utils(sp: Path) -> str:
    """Wrap _is_package_available so bare-bool callers see real booleans."""
    target = sp / "trl" / "import_utils.py"
    if not target.exists():
        return f"  trl/import_utils.py: not present (skip)"

    content = target.read_text(encoding="utf-8")
    if _TRL_MARKER in content:
        return "  trl/import_utils.py: already patched"

    needle = "from transformers.utils.import_utils import _is_package_available"
    if needle not in content:
        return "  trl/import_utils.py: needle not found — trl version may differ; manual review needed"

    replacement = (
        "from transformers.utils.import_utils import _is_package_available as _trnsf_is_package_available\n"
        "\n"
        "\n"
        "# Patch for transformers 5.x compat: _is_package_available now always returns\n"
        "# a (bool, str) tuple, but trl 0.24.0 assigns it to bare _<pkg>_available globals\n"
        "# and uses them as `if is_X_available():`. The non-empty tuple is always truthy,\n"
        "# so trl tries to import packages that aren't installed. This wrapper restores\n"
        "# the (return_version=False -> bool) contract trl expects.\n"
        "def _is_package_available(pkg_name, return_version=False):\n"
        "    r = _trnsf_is_package_available(pkg_name, return_version=return_version)\n"
        "    if isinstance(r, tuple) and not return_version:\n"
        "        return r[0]\n"
        "    return r"
    )
    target.write_text(content.replace(needle, replacement, 1), encoding="utf-8")
    return "  trl/import_utils.py: PATCHED"


def _patch_qwen3_5_modeling(sp: Path) -> str:
    """Add text-only fast-exit at the top of compute_3d_position_ids."""
    target = sp / "transformers" / "models" / "qwen3_5" / "modeling_qwen3_5.py"
    if not target.exists():
        return "  transformers/models/qwen3_5/modeling_qwen3_5.py: not present (skip)"

    content = target.read_text(encoding="utf-8")
    if _QWEN_MARKER in content:
        return "  transformers/models/qwen3_5/modeling_qwen3_5.py: already patched"

    needle = (
        "        mm_token_type_ids: torch.IntTensor | None = None,\n"
        "    ) -> torch.Tensor | None:\n"
        "        past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()"
    )
    if needle not in content:
        return (
            "  transformers/models/qwen3_5/modeling_qwen3_5.py: needle not found — "
            "transformers version may differ; manual review needed"
        )

    replacement = (
        "        mm_token_type_ids: torch.IntTensor | None = None,\n"
        "    ) -> torch.Tensor | None:\n"
        "        # Patch 2026-05-13 (Nova task #44): text-only fast-exit. When the caller\n"
        "        # provides no multimodal signal (image/video grid or mm_token_type_ids),\n"
        "        # skip 3D position-id computation entirely and let downstream derive\n"
        "        # positions from `cache_position`. Otherwise the reuse-cached-rope-deltas\n"
        "        # branch below (the `elif self.rope_deltas is not None` path) crashes\n"
        "        # with a shape mismatch on text-only training batches when rope_deltas\n"
        "        # was populated by an earlier multimodal call. Multimodal forward\n"
        "        # passes (with at least one of the three signals set) fall through to\n"
        "        # the original logic unchanged.\n"
        "        if image_grid_thw is None and video_grid_thw is None and mm_token_type_ids is None:\n"
        "            return None\n"
        "        past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()"
    )
    target.write_text(content.replace(needle, replacement, 1), encoding="utf-8")
    return "  transformers/models/qwen3_5/modeling_qwen3_5.py: PATCHED"


def _patch_llm_blender_file(target: Path) -> str:
    """TRANSFORMERS_CACHE → fallback to huggingface_hub.constants.HF_HUB_CACHE."""
    if not target.exists():
        return f"  {target.name}: not present (skip)"

    content = target.read_text(encoding="utf-8")
    if _BLENDER_MARKER in content:
        return f"  {target.name}: already patched"

    needle = "from transformers.utils.hub import TRANSFORMERS_CACHE"
    if needle not in content:
        return f"  {target.name}: needle not found — llm_blender version may differ; manual review needed"

    replacement = (
        "try:\n"
        "    from transformers.utils.hub import TRANSFORMERS_CACHE  # transformers < 5.0\n"
        "except ImportError:\n"
        "    from huggingface_hub.constants import HF_HUB_CACHE as TRANSFORMERS_CACHE  # transformers >= 5.0"
    )
    target.write_text(content.replace(needle, replacement, 1), encoding="utf-8")
    return f"  {target.name}: PATCHED"


def main() -> int:
    sp = _site_packages()
    print(f"finetune_env_compat_patches: applying to {sp}")

    results = [
        _patch_trl_import_utils(sp),
        _patch_llm_blender_file(sp / "llm_blender" / "blender" / "blender.py"),
        _patch_llm_blender_file(sp / "llm_blender" / "blender" / "blender_utils.py"),
        _patch_qwen3_5_modeling(sp),
    ]
    for line in results:
        print(line)

    # Quick verification pass.
    print("\nVerifying imports...")
    try:
        import unsloth  # noqa: F401
        from trl import DPOTrainer, GRPOTrainer  # noqa: F401
        from trl.import_utils import is_weave_available
        wv = is_weave_available()
        if not isinstance(wv, bool):
            print(f"  WARNING: is_weave_available returned {type(wv).__name__}; tuple-unwrap patch may not be active")
            return 1
        print(f"  trl trainers import OK; is_weave_available() returns proper bool ({wv})")
        return 0
    except Exception as e:
        print(f"  FAILED: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
