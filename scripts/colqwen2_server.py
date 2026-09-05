#!/usr/bin/env python3
"""Managed ColQwen2 dense embedding server for ClawGallery VDR.

Serves vidore/colqwen2-v1.0 via colpali-engine on CPU/CUDA. Speaks the same
POST /embed contract as the MLX server: ``embeddings`` is a list of
multi-vectors (one 128-d row per image patch / query token).

Requires: pip install colpali-engine torch transformers pillow
"""
from __future__ import annotations

import argparse
import importlib
import ipaddress
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

VALID_KINDS = {"image", "text", "caption"}
DEFAULT_MODEL = "vidore/colqwen2-v1.0"
DEFAULT_DIMENSIONS = 128
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".gguf"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="allow binding this unauthenticated local-file-reading server to a non-loopback host",
    )
    return parser.parse_args()


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    torch = importlib.import_module("torch")
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_loopback_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_bind_host(args: argparse.Namespace) -> None:
    if is_loopback_host(args.host) or args.allow_remote:
        return
    raise SystemExit(
        "error: refusing to bind unauthenticated /embed server to non-loopback host "
        f"{args.host!r} without --allow-remote; this server can read arbitrary "
        "local files requested by clients"
    )


def hf_hub_dir() -> Path:
    if cache := os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return Path(cache)
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return home / "hub"


def incomplete_hf_cache_message(model_name: str) -> str | None:
    root = hf_hub_dir() / f"models--{model_name.replace('/', '--')}"
    if not root.exists():
        return None
    snapshots = root / "snapshots"
    if not snapshots.is_dir():
        return (
            f"Hugging Face cache for {model_name} at {root} has no snapshots. "
            f"Delete {root} and retry. If downloads stall, toggle HF_HUB_DISABLE_XET "
            "(cdn-lfs vs xet) after upgrading huggingface_hub."
        )
    for snapshot in snapshots.iterdir():
        if not snapshot.is_dir():
            continue
        files = [path for path in snapshot.rglob("*") if path.is_file()]
        if any(path.suffix in WEIGHT_SUFFIXES for path in files):
            return None
        names = ", ".join(path.name for path in files[:8]) or "<empty>"
        return (
            f"Hugging Face snapshot for {model_name} is incomplete "
            f"(no model weights under {snapshot}; found {names}). "
            f"Delete {root} and retry. If downloads stall, toggle HF_HUB_DISABLE_XET "
            "(cdn-lfs vs xet) after upgrading huggingface_hub."
        )
    return None


def fake_multivector(value: str, dimensions: int) -> list[list[float]]:
    rows: list[list[float]] = []
    for word in re.split(r"[^a-z0-9]+", value.lower()):
        if not word:
            continue
        row = [0.0] * dimensions
        row[sum(ord(char) for char in word) % dimensions] = 1.0
        rows.append(row)
    return rows or [[0.0] * dimensions]


def load_colqwen(model_name: str, device: str):
    if message := incomplete_hf_cache_message(model_name):
        raise SystemExit(f"error: {message}")
    try:
        torch = importlib.import_module("torch")
        colpali = importlib.import_module("colpali_engine.models")
        model = colpali.ColQwen2.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
            device_map=device,
        ).eval()
        processor = colpali.ColQwen2Processor.from_pretrained(model_name)
        return torch, importlib.import_module("PIL.Image"), model, processor
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - surface Hub/runtime failures to the CLI
        extra = incomplete_hf_cache_message(model_name)
        hint = extra or (
            "If Hugging Face downloads stall, delete the incomplete cache, upgrade huggingface_hub, "
            "and toggle HF_HUB_DISABLE_XET (cdn-lfs vs xet)."
        )
        raise SystemExit(f"error: failed to load {model_name}: {exc}. {hint}") from exc


def make_server(model_name: str, dimensions: int, device: str, fake: bool) -> type[BaseHTTPRequestHandler]:
    loaded = None if fake else load_colqwen(model_name, device)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if self.path != "/embed":
                self.send_error(404, "not found")
                return
            if self.headers.get("origin") is not None:
                self.send_json(403, {"error": "browser-originated requests are not allowed"})
                return
            content_type = self.headers.get("content-type", "").partition(";")[0].strip().lower()
            if content_type != "application/json":
                self.send_json(415, {"error": "content-type must be application/json"})
                return
            try:
                length = int(self.headers.get("content-length", "0"))
                payload = json.loads(self.rfile.read(length))
            except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
                self.send_json(400, {"error": "request body must be valid JSON"})
                return
            if payload.get("model", model_name) != model_name:
                self.send_json(400, {"error": f"server loaded {model_name}"})
                return
            if payload.get("dimensions", dimensions) != dimensions:
                self.send_json(400, {"error": f"server loaded dimensions {dimensions}"})
                return
            try:
                self._embed(payload)
            except Exception as exc:  # noqa: BLE001 - keep /embed JSON-stable
                self.send_json(500, {"error": str(exc)})

        def _embed(self, payload: dict) -> None:
            if fake:
                embeddings = []
                for item in payload.get("inputs", []):
                    kind = item.get("kind")
                    if kind not in VALID_KINDS:
                        self.send_json(400, {"error": f"invalid input kind {kind!r}; expected image, text, or caption"})
                        return
                    value = str(item.get("value", ""))
                    haystack = Path(value).name if kind == "image" else value
                    embeddings.append(fake_multivector(haystack, dimensions))
                self.send_json(200, {"model": model_name, "dimensions": dimensions, "embeddings": embeddings})
                return

            assert loaded is not None
            torch, image_module, model, processor = loaded

            def to_multivectors(tensor):
                multivectors = []
                for doc in tensor.to(torch.float32).cpu():
                    rows = [row.tolist() for row in doc if float(row.abs().sum()) > 0.0]
                    if rows and len(rows[0]) != dimensions:
                        raise RuntimeError(
                            f"embedding server returned dimensions {len(rows[0])} but {dimensions} was requested"
                        )
                    multivectors.append(rows or [[0.0] * dimensions])
                return multivectors

            images, texts, order, opened = [], [], [], []
            try:
                for item in payload.get("inputs", []):
                    kind = item.get("kind")
                    if kind not in VALID_KINDS:
                        self.send_json(400, {"error": f"invalid input kind {kind!r}; expected image, text, or caption"})
                        return
                    if kind == "image":
                        image = image_module.open(Path(item["value"])).convert("RGB")
                        opened.append(image)
                        order.append(("image", len(images)))
                        images.append(image)
                    else:
                        order.append(("text", len(texts)))
                        texts.append(str(item.get("value", "")))
                image_vectors = []
                if images:
                    batch = processor.process_images(images).to(model.device)
                    with torch.no_grad():
                        image_vectors = to_multivectors(model(**batch))
                text_vectors = []
                if texts:
                    batch = processor.process_queries(texts).to(model.device)
                    with torch.no_grad():
                        text_vectors = to_multivectors(model(**batch))
                embeddings = [
                    image_vectors[index] if kind == "image" else text_vectors[index]
                    for kind, index in order
                ]
                self.send_json(200, {"model": model_name, "dimensions": dimensions, "embeddings": embeddings})
            finally:
                for image in opened:
                    image.close()

    return Handler


def main() -> None:
    args = parse_args()
    validate_bind_host(args)
    fake = os.environ.get("CLAWGALLERY_VDR_COLQWEN_FAKE") == "1"
    if not fake:
        if message := incomplete_hf_cache_message(args.model):
            raise SystemExit(f"error: {message}")
    device = "cpu" if fake else choose_device(args.device)
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_server(args.model, args.dimensions, device, fake),
    )
    print(
        json.dumps(
            {
                "url": f"http://{args.host}:{args.port}",
                "model": args.model,
                "dimensions": args.dimensions,
                "backend": "colqwen",
                "device": device,
                "fake": fake,
            }
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
