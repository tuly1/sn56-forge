"""Flight recorder: a small JSON written into the output directory.

The training container runs on the validator's hardware and its console is never
shown to us; the only artifact we reliably get back is the uploaded output
folder. So we keep a compact run log *inside* that folder (`forge_run.json`,
which the uploader does not filter out) and update it throughout the run. After
a tournament we can download our own uploaded repos and reconstruct exactly what
happened on each task — the post-mortem that makes week-over-week improvement
possible.

Design rules: never raise (a diagnostic must not cost a run), never grow beyond
a few hundred KB (curves are thinned), and never record anything sensitive
(nothing sensitive exists in the container anyway).
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

_FILENAME = "forge_run.json"
_MAX_EVENTS = 200
_MAX_CURVE_POINTS = 300
_MAX_SAMPLES = 120
_SENSITIVE_KEYS = {
    "authorization",
    "cookie",
    "credential",
    "credentials",
    "password",
    "passwd",
    "secret",
    "session",
    "signature",
    "token",
}
_SENSITIVE_COLLAPSED_KEYS = {
    "apikey",
    "accesskey",
    "authkey",
    "privatekey",
    "secretkey",
    "sessionid",
    "sessionkey",
    "sessiontoken",
}
_BEARER_RE = re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]+")
_AUTH_HEADER_RE = re.compile(
    r"(?i)\b((?:proxy-)?authorization\s*:\s*)"
    r"(?:(?:basic|bearer|digest)\s+)?[^\s,;]+"
)
_COOKIE_HEADER_RE = re.compile(r"(?im)^(\s*(?:set-)?cookie\s*:)[^\r\n]*")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b("
    r"api[_-]?key|access[_-]?key|auth[_-]?key|cookie|"
    r"aws[_-]?secret[_-]?access[_-]?key|client[_-]?secret|credentials?|"
    r"password|passwd|private[_-]?key|secret(?:[_-]?key)?|"
    r"(?:access|refresh|id)[_-]?token|"
    r"session(?:[_-]?(?:id|key|token))?|signature|token"
    r")\s*([:=])\s*([^\s,;&]+)"
)
_KNOWN_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"github_pat_[A-Za-z0-9_]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|"
    r"hf_[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16}"
    r")(?![A-Za-z0-9])"
)
_URL_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s'\"<>]+")

_t0 = time.monotonic()
_data: dict[str, Any] = {
    "schema": 1,
    "meta": {},
    "env": {},
    "events": [],
    "train_curve": [],  # (rel_s, step, loss, lr)
    "eval_curve": [],  # (rel_s, step, eval_loss)
    "samples": {},  # name -> [(rel_s, value)]
}


def _rel() -> float:
    return round(time.monotonic() - _t0, 1)


def init(**meta: Any) -> None:
    try:
        _data["meta"].update(
            {k: _sanitize_value(v, key=k) for k, v in meta.items() if v is not None}
        )
        _data["meta"].setdefault("started_unix", int(time.time()))
    except Exception:
        pass


def set_meta(**kv: Any) -> None:
    init(**kv)


def event(name: str, **kv: Any) -> None:
    try:
        if len(_data["events"]) >= _MAX_EVENTS:
            return
        safe = {k: _sanitize_value(v, key=k) for k, v in kv.items()}
        _data["events"].append({"t": _rel(), "name": name, **safe})
    except Exception:
        pass


def sample(name: str, value: float) -> None:
    try:
        series = _data["samples"].setdefault(name, [])
        if len(series) < _MAX_SAMPLES:
            series.append((_rel(), round(float(value), 6)))
    except Exception:
        pass


def train_point(step: int, loss: float | None, lr: float | None) -> None:
    try:
        curve = _data["train_curve"]
        curve.append((_rel(), int(step), loss, lr))
        if len(curve) > _MAX_CURVE_POINTS:
            # Thin by dropping every other point from the older half.
            half = len(curve) // 2
            _data["train_curve"] = curve[:half:2] + curve[half:]
    except Exception:
        pass


def eval_point(step: int, eval_loss: float) -> None:
    try:
        _data["eval_curve"].append((_rel(), int(step), round(float(eval_loss), 6)))
    except Exception:
        pass


def collect_env() -> None:
    """Versions + hardware; called from handlers once heavy imports exist."""
    try:
        import torch
        import transformers

        env = {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["gpu_mem_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            )
        try:
            import peft
            import trl

            env["trl"] = trl.__version__
            env["peft"] = peft.__version__
        except Exception:
            pass
        _data["env"].update(env)
    except Exception:
        pass


def note_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            _data["env"]["peak_cuda_alloc_gb"] = round(
                torch.cuda.max_memory_allocated() / 1e9, 2
            )
    except Exception:
        pass


def write_into(output_dir: str) -> None:
    """Atomically (tmp+replace) drop the log into the folder that gets uploaded."""
    try:
        if not os.path.isdir(output_dir):
            return
        _data["meta"]["last_write_rel_s"] = _rel()
        tmp = os.path.join(output_dir, _FILENAME + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(_data, fh, separators=(",", ":"), default=str)
        os.replace(tmp, os.path.join(output_dir, _FILENAME))
    except Exception:
        pass


def _sanitize_value(value: Any, *, key: str = "") -> Any:
    """Best-effort redaction for the public flight recorder.

    Tournament artifacts are public.  Diagnostics still need exception text and
    phase information, but must not publish bearer tokens, signed URL queries, or
    values stored under credential-like keys.  This helper deliberately keeps the
    URL path: a missing model/file path is useful forensic evidence, while its
    query string is where temporary credentials normally live.
    """
    low_key = key.lower()
    key_parts = [part for part in re.split(r"[^a-z0-9]+", low_key) if part]
    collapsed_key = "".join(key_parts)
    if any(part in _SENSITIVE_KEYS for part in key_parts) or any(
        marker in collapsed_key for marker in _SENSITIVE_COLLAPSED_KEYS
    ):
        return "<redacted>"
    if isinstance(value, dict):
        return {str(k): _sanitize_value(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(v, key=key) for v in value]
    if not isinstance(value, str):
        return value

    text = _COOKIE_HEADER_RE.sub(r"\1<redacted>", value)
    text = _AUTH_HEADER_RE.sub(r"\1<redacted>", text)
    text = _BEARER_RE.sub(r"\1<redacted>", text)

    def _strip_url(match: re.Match[str]) -> str:
        raw = match.group(0)
        # Punctuation commonly terminates a URL in an exception sentence.
        suffix = ""
        while raw and raw[-1] in ".,);]":
            suffix = raw[-1] + suffix
            raw = raw[:-1]
        try:
            parts = urlsplit(raw)
            netloc = parts.netloc
            if parts.username is not None or parts.password is not None:
                host = parts.hostname or ""
                if ":" in host and not host.startswith("["):
                    host = f"[{host}]"
                netloc = host
                if parts.port is not None:
                    netloc += f":{parts.port}"
            clean = urlunsplit((parts.scheme, netloc, parts.path, "", ""))
            return clean + suffix
        except Exception:
            return "<redacted-url>" + suffix

    # Strip complete URLs before redacting loose assignments.  Otherwise a
    # credential inside a query string can introduce ``<``/``>`` delimiters
    # that cause the URL matcher to stop midway through the same secret URL.
    text = _URL_RE.sub(_strip_url, text)
    text = _KNOWN_TOKEN_RE.sub("<redacted-token>", text)
    return _SECRET_ASSIGNMENT_RE.sub(r"\1\2<redacted>", text)


def make_trainer_callback(output_dir: str):
    """A TrainerCallback that records curves and periodically persists the log.

    Imported lazily so this module stays usable without transformers.
    """
    from transformers import TrainerCallback

    class TelemetryCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
            logs = logs or {}
            if "loss" in logs:
                train_point(state.global_step, logs.get("loss"), logs.get("learning_rate"))
            return control

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # noqa: ANN001
            loss = (metrics or {}).get("eval_loss")
            if loss is not None:
                eval_point(state.global_step, loss)
            write_into(output_dir)
            return control

        def on_train_end(self, args, state, control, **kwargs):  # noqa: ANN001
            # epochs completed is the single clearest "did we run out of time"
            # signal — capture it directly instead of deriving it post-hoc.
            event(
                "train_end",
                steps=int(state.global_step),
                epochs=round(float(getattr(state, "epoch", 0.0) or 0.0), 3),
                planned_epochs=round(
                    float(getattr(args, "num_train_epochs", 0.0) or 0.0), 3
                ),
            )
            note_peak_memory()
            write_into(output_dir)
            return control

    return TelemetryCallback()
