#!/usr/bin/env python3
"""Native Windows/CPU/CUDA V-SPLADE server for ClawGallery.

Set CLAWGALLERY_VSPLADE_REPO to a checkout of NAVER's V-SPLADE repository.
The server deliberately keeps the same POST /embed contract as the MLX server.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

VALID_KINDS = {"image", "text", "caption"}
DEFAULT_MODEL = "naver/v-splade-efficient"
DEFAULT_DIMENSIONS = 50368


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--allow-remote", action="store_true")
    return parser.parse_args()


def validate_bind_host(args: argparse.Namespace) -> None:
    if args.allow_remote or args.host == "localhost":
        return
    try:
        if ipaddress.ip_address(args.host).is_loopback:
            return
    except ValueError:
        pass
    raise SystemExit(
        f"error: refusing to bind unauthenticated /embed server to {args.host!r} "
        "without --allow-remote"
    )


def load_encoder(model_name: str, device_name: str):
    repo = os.environ.get("CLAWGALLERY_VSPLADE_REPO")
    if not repo:
        raise SystemExit(
            "error: CLAWGALLERY_VSPLADE_REPO must point to a V-SPLADE checkout; "
            "see the Windows setup instructions"
        )
    repo_path = Path(repo).resolve()
    inference_path = repo_path / "examples" / "vsplade_inference.py"
    if not inference_path.is_file():
        raise SystemExit(f"error: V-SPLADE inference helper not found at {inference_path}")
    sys.path.insert(0, str(repo_path))
    import importlib.util

    spec = importlib.util.spec_from_file_location("vsplade_inference", inference_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"error: could not load V-SPLADE inference helper at {inference_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    VSPLADEInference = module.VSPLADEInference

    import torch  # type: ignore[import-not-found]

    device = "cuda" if device_name in {"auto", "cuda"} and torch.cuda.is_available() else "cpu"
    return VSPLADEInference.from_pretrained(model_name, device=device, dtype=torch.float32)


def to_sparse(vector, dimensions: int) -> dict[str, list[float] | list[int]]:
    import torch  # type: ignore[import-not-found]

    dense = vector.detach().float().reshape(-1)
    if dense.numel() > dimensions:
        dense = dense[:dimensions]
    indices = torch.nonzero(dense > 0, as_tuple=False).reshape(-1)
    return {
        "indices": indices.cpu().tolist(),
        "values": dense[indices].cpu().tolist(),
    }


def make_server(model_name: str, dimensions: int, device_name: str) -> type[BaseHTTPRequestHandler]:
    fake = os.environ.get("CLAWGALLERY_VDR_VSPLADE_FAKE") == "1"
    encoder = None if fake else load_encoder(model_name, device_name)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_POST(self) -> None:
            if self.path != "/embed":
                self.send_error(404, "not found")
                return
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length))

            def send_json(status: int, body: dict) -> None:
                encoded = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            if payload.get("model", model_name) != model_name:
                send_json(400, {"error": f"server loaded {model_name}"})
                return
            if payload.get("dimensions", dimensions) != dimensions:
                send_json(400, {"error": f"server loaded dimensions {dimensions}"})
                return
            try:
                embeddings = []
                for item in payload.get("inputs", []):
                    kind = item.get("kind")
                    if kind not in VALID_KINDS:
                        send_json(400, {"error": f"invalid input kind {kind!r}"})
                        return
                    value = str(item.get("value") or "")
                    if fake:
                        haystack = (Path(value).name if kind == "image" else value).lower()
                        if "dog" in haystack or "puppy" in haystack:
                            index, weight = 1, 2.0
                        elif "cat" in haystack or "kitten" in haystack:
                            index, weight = 2, 2.0
                        else:
                            tokens = [token for token in re.split(r"[^a-z0-9]+", haystack) if token]
                            index, weight = sum(ord(char) for char in (tokens[0] if tokens else "x")) % max(dimensions, 1), 1.0
                        embeddings.append({"indices": [min(index, max(dimensions, 1) - 1)], "values": [weight]})
                    elif kind == "image":
                        from PIL import Image

                        assert encoder is not None
                        with Image.open(value) as image:
                            embeddings.append(to_sparse(encoder.encode_image(image), dimensions))
                    else:
                        assert encoder is not None
                        embeddings.append(to_sparse(encoder.encode_query(value), dimensions))
                send_json(
                    200,
                    {
                        "model": model_name,
                        "dimensions": dimensions,
                        "format": "sparse",
                        "embeddings": embeddings,
                    },
                )
            except Exception as exc:  # noqa: BLE001 - return actionable server errors
                send_json(500, {"error": str(exc)})

    return Handler


def main() -> None:
    args = parse_args()
    validate_bind_host(args)
    server = ThreadingHTTPServer((args.host, args.port), make_server(args.model, args.dimensions, args.device))
    server.serve_forever()


if __name__ == "__main__":
    main()
