"""Small OpenAI-compatible HTTP adapter for the v6 runtime."""

from __future__ import annotations

import json
import select
import socket
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import time
import uuid
from urllib.parse import parse_qs, urlparse

from core_models import backbone_ids, display_name, get_backbone

from .expert_library import LIBRARY
from .expert_packages import blocked, load_packages

from . import attachments, llm, main as runtime, runtime_selection, web_search
from .model_download import INSTALLER
from . import hub_experts
from .expert_catalog import EXPERT_CATALOG
from .expert_registry import EXPERT_REGISTRY
from .lora_runtime import ADAPTERS, refresh_package_adapters
from .main import run_request
from .request_journal import JOURNAL

MAX_BODY_BYTES = 2 * 1024 * 1024
# Chat requests may carry base64 images and files.
MAX_CHAT_BODY_BYTES = 64 * 1024 * 1024
# Arguments this server started with; a runtime switch restarts with the same options.
SERVER_ARGV: list[str] = []
# Why no base model is loaded (none downloaded yet); chat waits until one is chosen in the app.
SETUP_REASON: str | None = None
CLIENT_ROOT = Path(__file__).resolve().parents[1] / "client"
CLIENT_FILES = {
    "/app/": ("index.html", "text/html; charset=utf-8"),
    "/app/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/app/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app/vendor/katex/katex.min.js": ("vendor/katex/katex.min.js", "text/javascript; charset=utf-8"),
    "/app/vendor/katex/katex.min.css": ("vendor/katex/katex.min.css", "text/css; charset=utf-8"),
}
KATEX_FONT_DIR = CLIENT_ROOT / "vendor/katex/fonts"


def _tier(package: dict) -> dict:
    """Curated experts are routed automatically; adapters added from Hugging Face are not."""
    if package["profile"].get("manual_only"):
        return {"source": "huggingface", "manual_only": True, "verified": False}
    return {"source": "ghostswitch", "manual_only": False,
            "verified": (package.get("evaluation") or {}).get("verdict") == "helps"}


class Handler(BaseHTTPRequestHandler):
    def _client_name(self) -> str:
        # X-OmniBrain-Client and X-MiniBrain-Client are the headers from before the renames.
        name = (self.headers.get("X-GhostSwitch-Client") or self.headers.get("X-OmniBrain-Client")
                or self.headers.get("X-MiniBrain-Client") or "API")
        return name[:80]

    def _send(self, code: int, value: dict) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _current_runtime(self) -> dict:
        return {"backbone": llm.ACTIVE_BACKBONE, "name": display_name(llm.ACTIVE_BACKBONE),
                "backend": llm.MODEL_SPECS["thinker"].backend}

    def do_GET(self) -> None:
        if self.path == "/health":
            current = self._current_runtime()
            try:
                context_window = runtime._profile().get("context_window")
            except (OSError, ValueError):
                context_window = None
            self._send(200, {"status": "ok", "model": "ghostswitch-v6", "backend": current["backend"],
                             "backbone": current["backbone"], "backbone_name": current["name"],
                             "context_window": None if SETUP_REASON else context_window,
                             "setup_required": bool(SETUP_REASON), "setup_reason": SETUP_REASON,
                             "vision": not SETUP_REASON and llm.vision_available()})
        elif self.path == "/v1/search/settings":
            self._send(200, web_search.public_settings())
        elif self.path == "/v1/runtime":
            current = self._current_runtime()
            spec = llm.MODEL_SPECS["thinker"]
            # A switch restarts the process, so a loaded model's memory comes back.
            loaded = bool(llm.MODEL_MANAGER.loaded_model_ids())
            releasable = runtime_selection.memory_estimate_gb(
                current["backbone"], current["backend"],
                spec.model_id if current["backend"] == "gguf" else None) if loaded else 0
            overview = runtime_selection.overview()
            for row in overview["options"]:
                row["experts"] = len(load_packages(row["id"]))
            self._send(200, {"current": {**current, "loaded": loaded, "ready": not SETUP_REASON},
                             "setup_required": bool(SETUP_REASON), "setup_reason": SETUP_REASON,
                             **overview, "install": INSTALLER.status(),
                             "memory": {**overview["hardware"]["memory"], "releasable_gb": releasable or 0}})
        elif self.path == "/v1/runtime/install":
            self._send(200, {"install": INSTALLER.status()})
        elif self.path == "/v1/requests":
            self._send(200, {"object": "list", "data": JOURNAL.list()})
        elif self.path == "/v1/models":
            self._send(200, {"object": "list", "data": [
                {"id": "ghostswitch-v6", "object": "model", "created": 0,
                 "owned_by": "ghostswitch"}]})
        elif self.path == "/v1/experts":
            packages = load_packages(llm.ACTIVE_BACKBONE)
            # Blocked experts were measured as no better than the base model: they are not offered.
            experts = [EXPERT_REGISTRY.get(eid) for eid in sorted(packages)
                       if eid in ADAPTERS and not blocked(packages[eid])
                       and EXPERT_REGISTRY.check_backbone_compatibility(eid)]
            self._send(200, {"object": "list", "backend": llm.MODEL_SPECS["thinker"].backend,
                             "backbone": llm.ACTIVE_BACKBONE,
                             "data": [{"id": item.expert_id, "name": item.profile.name,
                                       "description": item.profile.description,
                                       "capabilities": list(item.profile.capabilities),
                                       "adapter": item.adapter_id,
                                       "default_enabled": packages[item.expert_id]["profile"].get("default_enabled", False),
                                       "source_url": "https://huggingface.co/" + item.adapter.repo_id,
                                       "revision": item.adapter.revision,
                                       "evaluation": packages[item.expert_id].get("evaluation"),
                                       **_tier(packages[item.expert_id]),
                                       **LIBRARY.status(item.expert_id)} for item in experts]})
        elif urlparse(self.path).path == "/v1/experts/hub":
            query = parse_qs(urlparse(self.path).query)
            needle = (query.get("q") or [""])[0].strip().lower()
            try:
                rows = hub_experts.discover(llm.ACTIVE_BACKBONE, refresh=(query.get("refresh") or [""])[0] == "1")
            except OSError as exc:
                self._send(502, {"error": {"message": f"Hugging Face is unreachable: {exc}"}})
                return
            matches = [row for row in rows if needle in row["repo_id"].lower()]
            limit = min(int((query.get("limit") or ["100"])[0]), 500)
            self._send(200, {"object": "list", "total": len(rows), "matches": len(matches),
                             "data": matches[:limit]})
        elif self.path in {"/", "/app"}:
            self.send_response(302)
            self.send_header("Location", "/app/")
            self.end_headers()
        elif self.path in CLIENT_FILES or self.path.startswith("/app/vendor/katex/fonts/"):
            asset = CLIENT_FILES.get(self.path)
            if asset is None:
                font_name = self.path.removeprefix("/app/vendor/katex/fonts/")
                font_path = KATEX_FONT_DIR / font_name
                if (font_path.name != font_name or font_path.suffix != ".woff2"
                        or not font_path.is_file()):
                    self._send(404, {"error": {"type": "not_found", "message": "Unknown path"}})
                    return
                asset = (str(font_path.relative_to(CLIENT_ROOT)), "font/woff2")
            name, content_type = asset
            content = (CLIENT_ROOT / name).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)
        else:
            self._send(404, {"error": {"type": "not_found", "message": "Unknown path"}})

    def _cross_origin(self) -> bool:
        # A remote web page must not change local settings through a browser.
        origin = self.headers.get("Origin")
        return bool(origin) and urlparse(origin).netloc != self.headers.get("Host")

    def _read_object(self) -> dict:
        size = int(self.headers.get("Content-Length", "0"))
        if not 1 <= size <= MAX_BODY_BYTES:
            raise ValueError("Invalid request body size")
        payload = json.loads(self.rfile.read(size))
        if not isinstance(payload, dict):
            raise ValueError("Request must be a JSON object")
        return payload

    def _switch_runtime(self) -> None:
        if self._cross_origin():
            self._send(403, {"error": {"message": "Cross-origin runtime changes are not allowed"}})
            return
        try:
            payload = self._read_object()
            backbone, backend = payload.get("backbone"), payload.get("backend")
            gguf_path = payload.get("gguf_path")
            if backbone not in backbone_ids() or backend not in runtime_selection.BACKENDS:
                raise ValueError("Choose a known backbone and mlx or gguf")
            if gguf_path is not None and (not isinstance(gguf_path, str) or not gguf_path.strip()):
                raise ValueError("gguf_path must be a file path")
            selection = runtime_selection.resolve(
                backbone, backend, str(Path(gguf_path).expanduser()) if gguf_path else None)
            # Fail here, not after the restart, if the checkpoint or GGUF file is unusable.
            get_backbone(selection.backbone, model_path=selection.model_path,
                         backend=selection.backend, validate_checkpoint=True)
        except (TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
            self._send(400, {"error": {"message": str(exc)}})
            return
        current = self._current_runtime()
        if not SETUP_REASON and (selection.backbone, selection.backend) == (current["backbone"], current["backend"]) and (
                selection.backend == "mlx" or selection.model_path == llm.MODEL_SPECS["thinker"].model_id):
            self._send(200, {"status": "current", **current})
            return
        if not runtime._RUN_LOCK.acquire(blocking=False):
            self._send(409, {"error": {"message": "A request is running; switch after it finishes"}})
            return
        runtime._RUN_LOCK.release()
        try:
            runtime_selection.save(selection)
        except OSError as exc:
            self._send(500, {"error": {"message": str(exc)}})
            return
        self._send(202, {"status": "restarting", "backbone": selection.backbone,
                         "name": display_name(selection.backbone), "backend": selection.backend})
        threading.Thread(target=_restart, args=(selection,), daemon=True).start()

    def _install_model(self) -> None:
        if self._cross_origin():
            self._send(403, {"error": {"message": "Cross-origin downloads are not allowed"}})
            return
        try:
            payload = self._read_object()
            if payload.get("action") == "cancel":
                self._send(200, {"install": INSTALLER.cancel()})
                return
            backbone, backend = payload.get("backbone"), payload.get("backend")
            if backbone not in backbone_ids() or backend not in runtime_selection.BACKENDS:
                raise ValueError("Choose a known backbone and mlx or gguf")
            gguf_file = runtime_selection.find_gguf(backbone) if backend == "gguf" else None
            if runtime_selection.installed(runtime_selection.Selection(
                    backbone, backend, str(gguf_file) if gguf_file else None)):
                self._send(200, {"install": {"backbone": backbone, "backend": backend, "status": "done",
                                             "model_path": str(gguf_file) if gguf_file else None}})
                return
            self._send(202, {"install": INSTALLER.start(backbone, backend)})
        except RuntimeError as exc:
            self._send(409, {"error": {"message": str(exc)}})
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send(400, {"error": {"message": str(exc)}})

    def _web_search(self) -> None:
        settings_change = self.path == "/v1/search/settings"
        if settings_change and self._cross_origin():
            self._send(403, {"error": {"message": "Cross-origin search settings changes are not allowed"}})
            return
        try:
            payload = self._read_object()
            if settings_change:
                self._send(200, web_search.save_settings(payload))
                return
            query = payload.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError("query must be a nonempty string")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send(400, {"error": {"message": str(exc)}})
            return
        except OSError as exc:
            self._send(500, {"error": {"message": str(exc)}})
            return
        result = web_search.search(query)
        body = result.public()
        # External tools receive the page text the model would see, marked as untrusted.
        for row, source in zip(body["sources"], result.sources):
            row["text"] = source.text
        self._send(502 if result.error else 200, body)

    def _hub_expert(self) -> None:
        """Add an adapter from Hugging Face as a manual-only expert, or remove one."""
        if self._cross_origin():
            self._send(403, {"error": {"message": "Cross-origin library changes are not allowed"}})
            return
        try:
            payload = self._read_object()
            if payload.get("action") == "add":
                expert_id = hub_experts.install(llm.ACTIVE_BACKBONE, str(payload.get("repo_id") or ""))
            elif payload.get("action") == "remove":
                expert_id = str(payload.get("expert_id") or "")
                if LIBRARY.enabled(expert_id):
                    LIBRARY.manage(expert_id, "disable")
                hub_experts.remove(llm.ACTIVE_BACKBONE, expert_id)
            else:
                raise ValueError("action must be add or remove")
            refresh_package_adapters(llm.ACTIVE_BACKBONE, llm.MODEL_SPECS["thinker"].model_id)
            EXPERT_CATALOG.reload()
            self._send(200, {"id": expert_id, "status": "added" if payload["action"] == "add" else "removed"})
        except KeyError:
            self._send(404, {"error": {"message": "Unknown added expert"}})
        except (TypeError, ValueError) as exc:
            self._send(400, {"error": {"message": str(exc)}})
        except OSError as exc:
            self._send(502, {"error": {"message": f"Hugging Face is unreachable: {exc}"}})

    def _manage_expert(self) -> None:
        if self._cross_origin():
            self._send(403, {"error": {"message": "Cross-origin library changes are not allowed"}})
            return
        expert_id = self.path.removeprefix("/v1/experts/")
        try:
            payload = self._read_object()
            result = LIBRARY.manage(expert_id, payload.get("action"))
            self._send(202 if result["status"] in {"queued", "downloading"} else 200,
                       {"id": expert_id, **result})
        except KeyError:
            self._send(404, {"error": {"message": "Unknown expert"}})
        except (TypeError, ValueError) as exc:
            self._send(400, {"error": {"message": str(exc)}})
        except OSError as exc:
            self._send(500, {"error": {"message": str(exc)}})

    def _client_disconnected(self) -> bool:
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            return bool(readable) and self.connection.recv(1, socket.MSG_PEEK) == b""
        except (OSError, ValueError):
            return True

    def do_POST(self) -> None:
        if self.path == "/v1/experts/hub":
            self._hub_expert()
            return
        if self.path.startswith("/v1/experts/"):
            self._manage_expert()
            return
        if self.path == "/v1/runtime":
            self._switch_runtime()
            return
        if self.path == "/v1/runtime/install":
            self._install_model()
            return
        if self.path in {"/v1/search", "/v1/search/settings"}:
            self._web_search()
            return
        if self.path != "/v1/chat/completions":
            self._send(404, {"error": {"type": "not_found", "message": "Unknown path"}})
            return
        if SETUP_REASON:
            self._send(503, {"status": "failed", "error": {
                "type": "model_not_installed",
                "message": f"No base model is loaded: {SETUP_REASON}. Open /app/ to download one."}})
            return
        completion_id = "chatcmpl-" + uuid.uuid4().hex
        payload = None
        logged = False
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size < 1 or size > MAX_CHAT_BODY_BYTES:
                raise ValueError(f"The request body must be between 1 byte and "
                                 f"{MAX_CHAT_BODY_BYTES // (1024 * 1024)} MB")
            payload = json.loads(self.rfile.read(size))
            # Images and files stay out of the journal; it keeps their type and size.
            JOURNAL.start(completion_id, attachments.redact(payload),
                          self._client_name())
            logged = True
            parsed = runtime.parse(payload)
            if isinstance(payload, dict) and payload.get("stream") is True:
                self._stream_chat_completion(payload, completion_id, parsed)
                return
            result = run_request(
                payload,
                parsed=parsed,
                completion_id=completion_id,
                is_cancelled=self._client_disconnected,
                on_event=lambda event: JOURNAL.update(completion_id, event=event)
                if event.get("type") in {"status", "stage", "progress"} else None,
            )
            result.setdefault("id", completion_id)
            JOURNAL.update(completion_id, result=result)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            result = {
                "id": completion_id,
                "status": "failed",
                "error": {"type": "invalid_request_error", "message": str(exc)},
            }
            if not logged:
                JOURNAL.start(
                    completion_id, attachments.redact(payload),
                    self._client_name(),
                )
            JOURNAL.update(completion_id, result=result)
            self._send(400, result)
            return
        code = 200 if result["status"] == "success" else (
            422 if result["status"] == "unsupported" else 500
        )
        self._send(code, result)

    def _stream_chat_completion(self, payload: dict, completion_id: str, parsed: dict) -> None:
        created = int(time.time())
        model = payload.get("model", "ghostswitch-v6")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def send_chunk(delta: dict, *, finish_reason: str | None = None,
                       extra: dict | None = None) -> None:
            chunk = {"id": completion_id, "object": "chat.completion.chunk",
                     "created": created, "model": model,
                     "choices": [{"index": 0, "delta": delta,
                                  "finish_reason": finish_reason}]}
            if extra:
                chunk.update(extra)
            data = json.dumps(chunk, ensure_ascii=False, separators=(",", ":"))
            self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
            self.wfile.flush()

        def on_event(event: dict) -> None:
            if event.get("type") in {"status", "stage", "progress"}:
                JOURNAL.update(completion_id, event=event)
            if payload.get("tools") and event["type"] == "status":
                self.log_message("agent stage=%s attempt=%s", event["status"], event.get("attempt", 0))
            if event["type"] == "token":
                send_chunk({"content": event["delta"]})
            elif event["type"] == "reasoning":
                send_chunk({"reasoning_content": event["delta"]})
            elif event["type"] == "accepted_message":
                message = event["message"]
                if message.get("reasoning_content"):
                    send_chunk({"reasoning_content": message["reasoning_content"]})
                if message.get("content"):
                    send_chunk({"content": message["content"]})
                for index, call in enumerate(message.get("tool_calls", [])):
                    send_chunk({"tool_calls": [{"index": index, **call}]})
            elif event["type"] == "reset":
                send_chunk({"content": "", "ghostswitch_reset": True,
                            "ghostswitch_attempt": event["attempt"]})
            elif event["type"] == "progress":
                send_chunk({}, extra={"ghostswitch_progress": {key: value for key, value in event.items() if key != "type"}})
            elif event["type"] == "status":
                send_chunk({"content": "", "ghostswitch_status": event["status"],
                            "ghostswitch_attempt": event.get("attempt")})

        try:
            send_chunk({"role": "assistant"})
            result = run_request(payload, on_event=on_event, completion_id=completion_id,
                                 is_cancelled=self._client_disconnected, parsed=parsed)
            JOURNAL.update(completion_id, result=result)
            finish_reason = result.get("choices", [{}])[0].get("finish_reason") if result.get("choices") else None
            send_chunk({}, finish_reason=finish_reason, extra={
                "usage": result.get("usage"), "status": result.get("status"),
                "ghostswitch": result.get("ghostswitch"), "routing": result.get("routing"),
                **({"error": result["error"]} if result.get("error") else {}),
            })
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # Keep an already completed result if only delivery disconnected.
            entries = JOURNAL.list()
            entry = next((item for item in entries if item["id"] == completion_id), None)
            if entry and entry["response"]["status"] in {"queued", "running"}:
                JOURNAL.update(completion_id, result={
                    "id": completion_id,
                    "status": "canceled",
                    "error": {
                        "type": "client_disconnected",
                        "message": "Client disconnected",
                    },
                })
            return
        except Exception as exc:
            result = {"status": "failed", "error": {"type": "inference_error", "message": str(exc)},
                      "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
            JOURNAL.update(completion_id, result=result)
            send_chunk({}, extra={key: result.get(key) for key in ("usage", "status", "error")})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()


def _restart(selection: runtime_selection.Selection) -> None:
    """Let the 202 response reach the client, wait for a running request, then restart."""
    time.sleep(0.5)
    with runtime._RUN_LOCK:
        runtime_selection.restart(SERVER_ARGV, selection)


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
