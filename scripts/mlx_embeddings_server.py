#!/usr/bin/env python3
import argparse
import importlib
import ipaddress
import json
import os
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Final, Literal, assert_never

VALID_KINDS: Final = {"image", "text", "caption"}
DEFAULT_MODEL: Final = "qnguyen3/colqwen2.5-v0.2-mlx"
EmbedKind = Literal["image", "text", "caption"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
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


def validate_bind_host(args: argparse.Namespace) -> None:
    if is_loopback_host(args.host) or args.allow_remote:
        return
    raise SystemExit(
        "error: refusing to bind unauthenticated /embed server to non-loopback host "
        f"{args.host!r} without --allow-remote; this server can read arbitrary "
        "local files requested by clients"
    )


def normalize_rows(rows: list[list[float]], dimensions: int) -> list[list[float]]:
    normalized: list[list[float]] = []
    for row in rows:
        trimmed = row[:dimensions]
        if len(trimmed) < dimensions:
            trimmed = [*trimmed, *([0.0] * (dimensions - len(trimmed)))]
        norm = sum(value * value for value in trimmed) ** 0.5
        if norm > 0.0:
            trimmed = [value / norm for value in trimmed]
        normalized.append(trimmed)
    return normalized or [[0.0] * dimensions]


def fake_multivector(value: str, dimensions: int) -> list[list[float]]:
    rows: list[list[float]] = []
    for word in re.split(r"[^a-z0-9]+", value.lower()):
        if not word:
            continue
        row = [0.0] * dimensions
        index = sum(ord(char) for char in word) % dimensions
        row[index] = 1.0
        rows.append(row)
    return rows or [[0.0] * dimensions]


def parse_kind(raw: str) -> EmbedKind | None:
    match raw:
        case "image" | "text" | "caption":
            return raw
        case _:
            return None


class Embedder:
    def __init__(self, model_name: str, dimensions: int, device: str) -> None:
        self.model_name = model_name
        self.dimensions = dimensions
        self.fake = os.environ.get("CLAWGALLERY_VDR_MLX_FAKE") == "1"
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        if not self.fake:
            mlx_embeddings = importlib.import_module("mlx_embeddings")
            transformers = importlib.import_module("transformers")
            self.model, self.tokenizer = mlx_embeddings.load(model_name)
            self.image_processor = transformers.AutoImageProcessor.from_pretrained(
                model_name,
                use_fast=False,
            )
            if device != "auto" and hasattr(self.model, "to"):
                self.model = self.model.to(device)

    def embed(self, kind: EmbedKind, value: str) -> list[list[float]]:
        match kind:
            case "text" | "caption":
                return self._embed_text(value)
            case "image":
                return self._embed_image(Path(value))
            case unreachable:
                assert_never(unreachable)

    def _embed_text(self, value: str) -> list[list[float]]:
        if self.fake:
            return fake_multivector(value, self.dimensions)
        return normalize_rows(self._embed_colqwen_text(value), self.dimensions)

    def _embed_image(self, path: Path) -> list[list[float]]:
        if self.fake:
            return fake_multivector(path.name, self.dimensions)
        return normalize_rows(self._embed_colqwen_image(path), self.dimensions)

    def _embed_colqwen_text(self, value: str) -> list[list[float]]:
        mx = importlib.import_module("mlx.core")
        base = importlib.import_module("mlx_embeddings.models.base")
        model = self._loaded_model()
        tokenizer = self._loaded_tokenizer()
        suffix = tokenizer.pad_token * 10
        encoded = tokenizer([f"Query: {value}{suffix}"], return_tensors="np", padding=True)
        input_ids = mx.array(encoded["input_ids"])
        attention_mask = mx.array(encoded["attention_mask"])
        inputs_embeds = model.get_input_embeddings_batch(input_ids)
        position_ids, _ = model.vlm.language_model.get_rope_index(
            input_ids,
            attention_mask=attention_mask,
        )
        hidden = model.vlm.language_model.model(
            None,
            inputs_embeds=inputs_embeds,
            mask=None,
            cache=None,
            position_ids=position_ids,
        )
        embeds = base.normalize_embeddings(model.embedding_proj_layer(hidden))
        embeds = embeds * attention_mask[:, :, None]
        return nonpad_rows(embeds, attention_mask)

    def _embed_colqwen_image(self, path: Path) -> list[list[float]]:
        mx = importlib.import_module("mlx.core")
        base = importlib.import_module("mlx_embeddings.models.base")
        image_module = importlib.import_module("PIL.Image")
        model = self._loaded_model()
        tokenizer = self._loaded_tokenizer()
        image_processor = self._loaded_image_processor()
        with image_module.open(path) as image:
            image_inputs = image_processor(
                images=[image.convert("RGB")],
                return_tensors="np",
                data_format="channels_first",
                do_convert_rgb=True,
            )
        image_grid_thw = mx.array(image_inputs["image_grid_thw"])
        num_image_tokens = int(
            image_inputs["image_grid_thw"][0].prod() // (image_processor.merge_size**2)
        )
        prompt = (
            "<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>"
            "Describe the image.<|im_end|><|endoftext|>"
        ).replace("<|image_pad|>", "<|image_pad|>" * num_image_tokens)
        text_inputs = tokenizer([prompt], return_tensors="np", padding=True)
        input_ids = mx.array(text_inputs["input_ids"])
        attention_mask = mx.array(text_inputs["attention_mask"])
        pixel_values = mx.array(image_inputs["pixel_values"])
        inputs_embeds = model.get_input_embeddings_batch(
            input_ids,
            pixel_values,
            image_grid_thw,
        )
        position_ids, _ = model.vlm.language_model.get_rope_index(
            input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )
        hidden = model.vlm.language_model.model(
            None,
            inputs_embeds=inputs_embeds,
            mask=None,
            cache=None,
            position_ids=position_ids,
        )
        embeds = base.normalize_embeddings(model.embedding_proj_layer(hidden))
        embeds = embeds * attention_mask[:, :, None]
        return nonpad_rows(embeds, attention_mask)

    def _loaded_model(self):
        if self.model is None:
            raise RuntimeError("mlx embedding model is not loaded")
        return self.model

    def _loaded_tokenizer(self):
        if self.tokenizer is None:
            raise RuntimeError("mlx embedding tokenizer is not loaded")
        return self.tokenizer

    def _loaded_image_processor(self):
        if self.image_processor is None:
            raise RuntimeError("mlx image processor is not loaded")
        return self.image_processor


def nonpad_rows(embeds, attention_mask) -> list[list[float]]:
    indices = [index for index, value in enumerate(attention_mask[0].tolist()) if value != 0]
    rows = embeds[0, indices, :]
    if hasattr(rows, "tolist"):
        rows = rows.tolist()
    return [[float(value) for value in row] for row in rows]


def make_handler(embedder: Embedder) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/embed":
                self.send_error(404, "not found")
                return
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length))
            embeddings: list[list[list[float]]] = []
            for item in payload.get("inputs", []):
                kind = parse_kind(str(item.get("kind")))
                if kind is None:
                    self._send_json(
                        400,
                        {"error": f"invalid input kind {item.get('kind')!r}; expected image, text, or caption"},
                    )
                    return
                embeddings.append(embedder.embed(kind, str(item.get("value", ""))))
            self._send_json(
                200,
                {
                    "model": embedder.model_name,
                    "dimensions": embedder.dimensions,
                    "embeddings": embeddings,
                },
            )

        def _send_json(self, status: int, payload: dict[str, object]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def main() -> None:
    args = parse_args()
    validate_bind_host(args)
    embedder = Embedder(args.model, args.dimensions, args.device)
    server = HTTPServer((args.host, args.port), make_handler(embedder))
    print(
        json.dumps(
            {
                "url": f"http://{args.host}:{args.port}",
                "model": args.model,
                "dimensions": args.dimensions,
                "backend": "mlx",
                "fake": embedder.fake,
            }
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
