#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "av",
#   "huggingface-hub",
#   "librosa",
#   "mlx>=0.23",
#   "pillow",
#   "requests",
#   "tokenizers",
#   "torch",
#   "torchvision",
#   "transformers>=4.57,<5",
# ]
# ///
# ─── How to run ───
# uv run scripts/jina_mlx_embeddings_server.py --help

import argparse
import importlib
import ipaddress
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Final, Literal, assert_never

MODEL_ID: Final = "jinaai/jina-embeddings-v5-omni-small-retrieval-mlx"
MODEL_REVISION: Final = "049ae923674456656be891ebb22849dd58124994"
DIMENSIONS: Final = 1024
VISION_PROMPT: Final = "<|vision_start|><|image_pad|><|vision_end|>"
MAX_BODY_BYTES: Final = 1_048_576
MAX_INPUTS: Final = 32
MAX_TEXT_CHARS: Final = 32_768
MAX_IMAGE_BYTES: Final = 52_428_800
REQUEST_TIMEOUT_SECONDS: Final = 30
EmbedKind = Literal["image", "text", "caption"]
EmbedRole = Literal["query", "document"]
JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
KINDS: Final[dict[str, EmbedKind]] = {"image": "image", "text": "text", "caption": "caption"}
ROLES: Final[dict[str, EmbedRole]] = {"query": "query", "document": "document"}


class InvalidEmbeddingError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RequestError(Exception):
    status: int
    message: str

    def __str__(self) -> str:
        return self.message


EmbeddingInput = tuple[EmbedKind, EmbedRole, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--dimensions", type=int, default=DIMENSIONS)
    parser.add_argument("--device", default="auto", choices=["auto", "mps"])
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="allow binding this unauthenticated local-file-reading server to a non-loopback host",
    )
    return parser.parse_args()


def is_loopback_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_args(args: argparse.Namespace) -> None:
    if args.model != MODEL_ID:
        raise SystemExit(f"error: jina-mlx requires model {MODEL_ID}")
    if args.dimensions != DIMENSIONS:
        raise SystemExit(f"error: jina-mlx requires {DIMENSIONS} dimensions")
    if not (is_loopback_host(args.host) or args.allow_remote):
        raise SystemExit(
            "error: refusing to bind unauthenticated /embed server to non-loopback host "
            f"{args.host!r} without --allow-remote; this server can read arbitrary "
            "local files requested by clients"
        )


def fake_vector(value: str, role: EmbedRole) -> list[float]:
    vector = [0.0] * DIMENSIONS
    index = sum(ord(char) for char in f"{role}:{value}") % DIMENSIONS
    vector[index] = 1.0
    return vector


class Embedder:
    def __init__(self) -> None:
        self.fake = os.environ.get("CLAWGALLERY_VDR_JINA_MLX_FAKE") == "1"
        self.mx = None
        self.model = None
        self.tokenizer = None
        self.processor = None
        if self.fake:
            return
        huggingface_hub = importlib.import_module("huggingface_hub")
        tokenizers = importlib.import_module("tokenizers")
        self.mx = importlib.import_module("mlx.core")
        snapshot = Path(
            huggingface_hub.snapshot_download(
                repo_id=MODEL_ID,
                revision=MODEL_REVISION,
            )
        )
        sys.path.insert(0, str(snapshot))
        model_module = importlib.import_module("model")
        config = model_module.OmniSmallConfig.from_dict(
            json.loads((snapshot / "config.json").read_text())
        )
        self.model = model_module.JinaOmniSmallEmbeddingModel(config)
        weights = self.model.sanitize(self.mx.load(str(snapshot / "model.safetensors")))
        self.model.load_weights(list(weights.items()))
        self.mx.eval(self.model.parameters())
        self.tokenizer = tokenizers.Tokenizer.from_file(str(snapshot / "tokenizer.json"))
        transformers = importlib.import_module("transformers")
        self.processor = transformers.AutoProcessor.from_pretrained(
            str(snapshot),
            trust_remote_code=True,
        )

    def embed(self, kind: EmbedKind, role: EmbedRole, value: str) -> list[float]:
        if len(value) > MAX_TEXT_CHARS:
            raise RequestError(413, f"input value exceeds {MAX_TEXT_CHARS} characters")
        if kind == "image":
            path = Path(value)
            try:
                image_size = path.stat().st_size
            except OSError as error:
                raise RequestError(400, "image path is not readable") from error
            if not path.is_file():
                raise RequestError(400, "image path must reference a regular file")
            if image_size > MAX_IMAGE_BYTES:
                raise RequestError(413, f"image exceeds {MAX_IMAGE_BYTES} bytes")
        if self.fake:
            return fake_vector(value, role)
        match kind:
            case "text" | "caption":
                return self._embed_text(role, value)
            case "image":
                return self._embed_image(Path(value))
            case unreachable:
                assert_never(unreachable)

    def _embed_text(self, role: EmbedRole, value: str) -> list[float]:
        prefix = "Query: " if role == "query" else "Document: "
        encoded = self.tokenizer.encode(f"{prefix}{value}")
        input_ids = self.mx.array([encoded.ids])
        attention_mask = self.mx.array([encoded.attention_mask])
        return self._vector(self.model.encode_text(input_ids, attention_mask))

    def _embed_image(self, path: Path) -> list[float]:
        image_module = importlib.import_module("PIL.Image")
        with image_module.open(path) as image:
            inputs = self.processor(
                images=[image.convert("RGB")],
                text=VISION_PROMPT,
                return_tensors="pt",
                truncation=False,
            )
        embedding = self.model.encode_image(
            self.mx.array(inputs["pixel_values"].numpy()),
            self.mx.array(inputs["image_grid_thw"].numpy()),
            self.mx.array(inputs["input_ids"].numpy()),
            self.mx.array(inputs["attention_mask"].numpy()),
        )
        return self._vector(embedding)

    def _vector(self, embedding) -> list[float]:
        values = [float(value) for value in embedding[0].tolist()]
        if len(values) != DIMENSIONS:
            raise InvalidEmbeddingError(
                f"Jina MLX returned {len(values)} dimensions, expected {DIMENSIONS}"
            )
        norm = math.sqrt(sum(value * value for value in values))
        if not math.isfinite(norm) or norm == 0.0:
            raise InvalidEmbeddingError("Jina MLX returned a non-finite or zero vector")
        return [value / norm for value in values]


def parse_inputs(payload: JsonValue) -> list[EmbeddingInput]:
    match payload:  # noqa: MATCH_OK — untrusted JSON must reject unknown shapes
        case {"inputs": list(raw_inputs)}:
            if len(raw_inputs) > MAX_INPUTS:
                raise RequestError(413, f"request exceeds {MAX_INPUTS} inputs")
        case _:
            raise RequestError(400, "request body must contain an inputs array")

    inputs: list[EmbeddingInput] = []
    for raw_input in raw_inputs:
        match raw_input:  # noqa: MATCH_OK — untrusted JSON must reject unknown shapes
            case {"kind": str(raw_kind), "role": str(raw_role), "value": str(value)}:
                kind = KINDS.get(raw_kind)
                if kind is None:
                    raise RequestError(400, "kind must be image, text, or caption")
                role = ROLES.get(raw_role)
                if role is None:
                    raise RequestError(400, "role must be query or document")
                inputs.append((kind, role, value))
            case _:
                raise RequestError(400, "each input must contain string kind, role, and value")
    return inputs


def make_handler(embedder: Embedder) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            try:
                self._do_post()
            except RequestError as error:
                self._send_json(error.status, {"error": error.message})
            except (InvalidEmbeddingError, RuntimeError, OSError):
                traceback.print_exc(file=sys.stderr)
                self._send_json(500, {"error": "embedding failed"})

        def _do_post(self) -> None:
            if self.path != "/embed":
                raise RequestError(404, "not found")
            if self.headers.get("origin") is not None:
                raise RequestError(403, "browser-originated requests are not allowed")
            content_type = self.headers.get("content-type", "").partition(";")[0].strip().lower()
            if content_type != "application/json":
                raise RequestError(415, "content-type must be application/json")
            try:
                length = int(self.headers.get("content-length", ""))
            except ValueError as error:
                raise RequestError(400, "content-length must be an integer") from error
            if length <= 0:
                raise RequestError(400, "request body must not be empty")
            if length > MAX_BODY_BYTES:
                raise RequestError(413, f"request body exceeds {MAX_BODY_BYTES} bytes")
            self.connection.settimeout(REQUEST_TIMEOUT_SECONDS)
            raw_body = self.rfile.read(length)
            if len(raw_body) != length:
                raise RequestError(400, "request body ended before content-length")
            try:
                payload: JsonValue = json.loads(raw_body)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise RequestError(400, "request body must be valid JSON") from error
            embeddings = [embedder.embed(*item) for item in parse_inputs(payload)]
            self._send_json(
                200,
                {"model": MODEL_ID, "dimensions": DIMENSIONS, "embeddings": embeddings},
            )

        def _send_json(self, status: int, payload: dict[str, JsonValue]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: str | int | float) -> None:
            return

    return Handler


def main() -> None:
    args = parse_args()
    validate_args(args)
    embedder = Embedder()
    server = HTTPServer((args.host, args.port), make_handler(embedder))
    print(
        json.dumps(
            {
                "url": f"http://{args.host}:{args.port}",
                "model": MODEL_ID,
                "revision": MODEL_REVISION,
                "dimensions": DIMENSIONS,
                "backend": "jina-mlx",
                "fake": embedder.fake,
            }
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
