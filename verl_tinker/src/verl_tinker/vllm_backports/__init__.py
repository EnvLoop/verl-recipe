# Copyright 2026 EnvLoop
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Upstream vLLM fixes the rollout needs, backported onto the locked vLLM and applied at server start.

* ``qwen3_5-lm-head-lora-<version>.diff``: vLLM PR #48850 (``5a4c8d99242e``, in vLLM since 0.27). Without
  ``embedding_modules`` on ``Qwen3_5ForCausalLMBase`` vLLM never wraps the output layer, and VeRL's in-memory
  adapter sync (``TensorLoRARequest``) does not check expected modules, so an ``lm_head`` LoRA (Tinker's
  ``train_unembed``) was silently dropped by the rollout.
* ``lora-prompt-logprobs-<version>.diff``: vLLM PR #62049 (fixes vllm-project/vllm#51594, open upstream).
  ``LogitsProcessorWithLoRA`` applies the ``lm_head`` delta with the per-request sampling mapping while prompt
  logprobs compute logits for every prompt token, so a request's own ``lm_head`` adapter is missing from its
  prompt logprobs and, in mixed batches, other requests' adapters leak in. Same diff the EnvLoop sampling pool
  applies (EnvPlatform ``env_platform/serving/guest/vllm_patches``).

The applier (from EnvPlatform ``patch_lora_prompt_logprobs.py``) is stdlib only, idempotent, and refuses to
start on a vLLM tree that matches neither the unpatched nor the patched code. Remove a diff once the locked
vLLM contains its fix.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
import pathlib
import re

logger = logging.getLogger(__name__)

PATCHES = pathlib.Path(__file__).parent
FIXES = ("qwen3_5-lm-head-lora", "lora-prompt-logprobs")


class BackportError(RuntimeError):
    pass


def hunks(diff: str) -> dict[str, list[tuple[str, str]]]:
    """(path, [(old_block, new_block), ...]) per file of a unified diff; hunk sizes come from the @@ headers."""
    files, path, old_left, new_left, current = {}, None, 0, 0, None
    for line in diff.splitlines(keepends=True):
        if old_left or new_left:
            kind, text = (line[0], line[1:]) if line[:1] in (" ", "-", "+") else (" ", line)  # blank context
            if kind in (" ", "-"):
                current[0].append(text)
                old_left -= 1
            if kind in (" ", "+"):
                current[1].append(text)
                new_left -= 1
        elif line.startswith("+++ "):
            path = line[4:].split("\t")[0].strip().removeprefix("b/")
            files[path] = []
        elif line.startswith("@@"):
            sizes = re.match(r"@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@", line)
            old_left, new_left = int(sizes.group(1) or 1), int(sizes.group(2) or 1)
            current = [[], []]
            files[path].append(current)
    return {p: [("".join(o), "".join(n)) for o, n in hs] for p, hs in files.items()}


def apply(root: pathlib.Path, version: str, fixes=FIXES) -> list[str]:
    """Apply the backports for ``version`` to the vLLM package at ``root``; returns the changed files."""
    changed = []
    for fix in fixes:
        diff = PATCHES / f"{fix}-{version}.diff"
        if not diff.exists():
            continue  # the vLLM release already contains the fix, or it is not a version we run
        for relative, file_hunks in hunks(diff.read_text()).items():
            target = root.parent / relative  # paths start with vllm/
            source = original = target.read_text()
            for old, new in file_hunks:
                if source.count(old) == 1:
                    source = source.replace(old, new)
                elif source.count(new) != 1:  # neither unpatched nor already patched
                    raise BackportError(f"{fix} backport does not match {relative}")
            if source != original:
                target.write_text(source)
                changed.append(relative)
    return changed


def _installed_root_and_version():
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        return None, None
    return pathlib.Path(spec.origin).parent, re.sub(r"\+.*$", "", importlib.metadata.version("vllm"))


def present(root: pathlib.Path, version: str, fixes=FIXES) -> bool:
    """Every fix is in the vLLM tree: its backport is applied, or the release ships it (no diff)."""
    for fix in fixes:
        diff = PATCHES / f"{fix}-{version}.diff"
        if not diff.exists():
            continue
        for relative, file_hunks in hunks(diff.read_text()).items():
            source = (root.parent / relative).read_text()
            if any(source.count(new) != 1 for _, new in file_hunks):
                return False
    return True


def find_installed() -> bool:
    return installed_root() is not None


def installed_root() -> pathlib.Path | None:
    """The installed ``vllm`` package directory, without importing vLLM."""
    return _installed_root_and_version()[0]


def present_installed() -> bool:
    """``present`` for the installed vLLM (False without vLLM). Reads files only; works in any process."""
    root, version = _installed_root_and_version()
    return root is not None and present(root, version)


def apply_installed() -> list[str]:
    """Patch the installed vLLM before any vLLM worker process imports it (no-op without vLLM)."""
    root, version = _installed_root_and_version()
    if root is None:
        return []
    changed = apply(root, version)
    logger.info("vLLM %s backports applied: %s", version, ", ".join(changed) or "(already applied)")
    return changed
