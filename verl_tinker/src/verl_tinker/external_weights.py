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

"""Optional external export handler, scheduled by the existing exclusive FIFO.

The deployment owns the handler and its storage/encryption policy. This module
does not implement a second engine, call the optimizer, or synchronize rollout.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import re
import uuid
from collections.abc import Mapping

from fastapi import HTTPException

MAX_EXTERNAL_REQUESTS = 10000


def _error(status, code, message):
    return HTTPException(status, {"code": code, "message": message})


def _handler(server):
    configured = server._server_option("external_weights_handler", None)
    if not isinstance(configured, str) or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+", configured):
        raise _error(501, "EXTERNAL_WEIGHTS_UNSUPPORTED", "No external weights export handler is configured")
    module, name = configured.rsplit(".", 1)
    try:
        handler = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError):
        raise _error(503, "EXTERNAL_WEIGHTS_UNAVAILABLE", "Configured external export handler is unavailable") from None
    if not callable(handler):
        raise _error(503, "EXTERNAL_WEIGHTS_UNAVAILABLE", "Configured external export handler is not callable")
    return handler


def _validate(server, body):
    fields = {"model_id", "path", "seq_id", "ttl_seconds", "age_encryption_recipients", "type"}
    if not isinstance(body, Mapping) or set(body) - fields:
        raise _error(400, "INVALID_EXTERNAL_SAVE", "Invalid external weights save request")
    if body.get("type", "save_weights_external") != "save_weights_external":
        raise _error(400, "INVALID_EXTERNAL_SAVE", "Invalid external weights request type")
    if not isinstance(body.get("model_id"), str):
        raise _error(400, "INVALID_MODEL_ID", "A training model id is required")
    if body["model_id"] not in server._model_to_base_model:
        raise _error(404, "MODEL_NOT_FOUND", "Unknown training model")
    if not server._lora_info().is_lora:
        raise _error(400, "EXTERNAL_WEIGHTS_UNSUPPORTED", "External adapter export requires a LoRA model")
    name = body.get("path")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]{0,127}", name):
        raise _error(400, "INVALID_CHECKPOINT_NAME", "External weights name must be one safe path component")
    seq = body.get("seq_id")
    if seq is not None and (type(seq) is not int or seq < 0):
        raise _error(400, "INVALID_SEQUENCE_ID", "seq_id must be a non-negative integer")
    ttl = body.get("ttl_seconds")
    if ttl is not None and (type(ttl) is not int or not 3600 <= ttl <= 3653 * 86400):
        raise _error(400, "INVALID_CHECKPOINT_TTL", "ttl_seconds must be one hour through ten years, or null")
    recipients = body.get("age_encryption_recipients")
    if recipients is not None and (
        not isinstance(recipients, list)
        or not 1 <= len(recipients) <= 10
        or any(not isinstance(key, str) or not key or len(key) > 16384 for key in recipients)
    ):
        raise _error(400, "INVALID_AGE_RECIPIENTS", "Provide one to ten public recipients, or null")


async def submit_external_weights(server, body):
    """Validate synchronously, then enqueue exactly one immutable export operation."""
    server._require_ready()
    _validate(server, body)
    handler = _handler(server)
    payload = json.loads(json.dumps(dict(body), allow_nan=False))
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    seq = body.get("seq_id")
    key = (body["model_id"], "seq", seq) if seq is not None else (body["model_id"], "name", body["path"])
    registry = server.__dict__.setdefault("_external_weight_requests", {})
    if key in registry:
        previous = registry[key]
        if previous["digest"] != fingerprint:
            raise _error(
                409, "EXTERNAL_SAVE_CONFLICT", "External save sequence was already accepted with different input"
            )
        return server._future_envelope(previous["request_id"], model_id=body["model_id"])
    if len(registry) >= MAX_EXTERNAL_REQUESTS:
        raise _error(429, "EXTERNAL_SAVE_LIMIT", "External export request limit reached")

    request_id = "external-" + uuid.uuid4().hex
    record = {"request_id": request_id, "digest": fingerprint}
    registry[key] = record
    server.__dict__.setdefault("_external_weight_results", {})[request_id] = record

    async def export():
        try:
            result = await asyncio.to_thread(
                handler, server._get_engine(), server.config, server._checkpoint_root, payload
            )
            if (
                not isinstance(result, dict)
                or result.get("type") != "save_weights_external"
                or not isinstance(result.get("path"), str)
            ):
                raise ValueError("invalid external export handler result")
            record["result"] = result
            return result
        except Exception as error:
            # Handler messages can contain filesystem/provider details; do not surface them.
            record["result"] = {
                "error": "External weights export failed",
                "category": "Application",
                "code": getattr(error, "code", "EXTERNAL_EXPORT_FAILED"),
            }
            raise

    try:
        # Default kind is EXCLUSIVE: preserves the same FIFO as forward/backward,
        # optim_step, load and save, without a separate writer or optimizer call.
        server._schedule(request_id, export())
    except BaseException:
        registry.pop(key, None)
        server._external_weight_results.pop(request_id, None)
        raise
    return server._future_envelope(request_id, model_id=body["model_id"])


def retained_external_result(server, request_id):
    """External save retries remain readable after the generic recent-result LRU evicts them."""
    record = server.__dict__.get("_external_weight_results", {}).get(request_id)
    if record is None or request_id in server._pending:
        return None
    return record.get("result")
