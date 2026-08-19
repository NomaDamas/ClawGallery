#!/usr/bin/env python3
"""V-SPLADE sparse lexical embedding server for ClawGallery.

Speaks the ClawGallery POST /embed contract, but returns sparse postings:
``{"indices":[int,...],"values":[float,...]}`` per input instead of dense
multi-vectors. Document images are encoded with the MLX V-SPLADE encoder;
queries use the inference-free Li-LSR lookup table.

Requires a Python env with splade-mlx, mlx, transformers, numpy, and pillow.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

VALID_KINDS = {"image", "text", "caption"}
DEFAULT_MODEL = "NomaDamas/v-splade-efficient-mlx"
DEFAULT_DIMENSIONS = 50368
DOC_PROMPT = "User:<image><end_of_utterance>\nAssistant:"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="allow binding this unauthenticated local-file-reading server to a non-loopback host",
    )
    return parser.parse_args()


def is_loopback_host(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_bind_host(args):
    if is_loopback_host(args.host) or args.allow_remote:
        return
    raise SystemExit(
        "error: refusing to bind unauthenticated /embed server to non-loopback host "
        f"{args.host!r} without --allow-remote; this server can read arbitrary "
        "local files requested by clients"
    )


def fake_sparse(value, dimensions):
    haystack = value.lower()
    if "dog" in haystack or "puppy" in haystack:
        index = 1
        weight = 2.0
    elif "cat" in haystack or "kitten" in haystack:
        index = 2
        weight = 2.0
    elif "login" in haystack:
        index = 3
        weight = 2.0
    else:
        tokens = [tok for tok in re.split(r"[^a-z0-9]+", haystack) if tok]
        index = sum(ord(char) for char in (tokens[0] if tokens else "x")) % max(dimensions, 1)
        weight = 1.0
    index = min(index, max(dimensions, 1) - 1)
    return {"indices": [index], "values": [weight]}


def to_sparse(vector, dimensions):
    import numpy as np

    dense = np.asarray(vector, dtype=np.float32).reshape(-1)
    if dense.size > dimensions:
        dense = dense[:dimensions]
    mask = dense > 0.0
    indices = np.nonzero(mask)[0]
    return {
        "indices": indices.astype(int).tolist(),
        "values": dense[indices].astype(float).tolist(),
    }


def make_server(model_name, dimensions, _device):
    fake = os.environ.get("CLAWGALLERY_VDR_VSPLADE_FAKE") == "1"
    model = query_encoder = processor = None
    if not fake:
        import mlx.core as mx
        import numpy as np
        from PIL import Image
        from splade_mlx.convert_vsplade import load_vsplade

        model, query_encoder, processor = load_vsplade(model_name)
        vocab = int(query_encoder.num_dimensions)
        if dimensions != vocab:
            raise SystemExit(
                f"error: V-SPLADE model {model_name} has vocabulary {vocab}, "
                f"but --dimensions {dimensions} was requested"
            )

        def encode_image(path):
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                enc = processor(
                    text=[DOC_PROMPT],
                    images=[[rgb]],
                    return_tensors="np",
                )
            pixel_values = enc["pixel_values"].astype("float32")
            sparse = model.encode(
                mx.array(enc["input_ids"]),
                mx.array(enc["attention_mask"]),
                pixel_values,
            )
            mx.eval(sparse)
            return to_sparse(np.array(sparse.astype(mx.float32))[0], dimensions)

        def encode_query(text):
            encoded = processor.tokenizer([text], return_tensors="np")
            weights = query_encoder.encode(encoded["input_ids"], encoded["attention_mask"])
            return to_sparse(weights[0], dimensions)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_POST(self):
            if self.path != "/embed":
                self.send_error(404, "not found")
                return
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length))

            def send_json(status, body):
                encoded = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            requested_model = payload.get("model") or model_name
            if requested_model != model_name:
                send_json(
                    400,
                    {
                        "error": (
                            f"server loaded {model_name} but client requested {requested_model}"
                        )
                    },
                )
                return
            requested_dimensions = payload.get("dimensions", dimensions)
            if requested_dimensions != dimensions:
                send_json(
                    400,
                    {
                        "error": (
                            f"server loaded dimensions {dimensions} but client "
                            f"requested {requested_dimensions}"
                        )
                    },
                )
                return

            embeddings = []
            try:
                for item in payload.get("inputs", []):
                    kind = item.get("kind")
                    if kind not in VALID_KINDS:
                        send_json(
                            400,
                            {
                                "error": (
                                    f"invalid input kind {kind!r}; expected image, text, or caption"
                                )
                            },
                        )
                        return
                    value = str(item.get("value") or "")
                    role = str(item.get("role") or "")
                    if fake:
                        haystack = Path(value).name if kind == "image" else value
                        embeddings.append(fake_sparse(haystack, dimensions))
                        continue
                    if kind == "image" or (kind == "caption" and role == "document"):
                        embeddings.append(encode_image(value) if kind == "image" else encode_query(value))
                    else:
                        embeddings.append(encode_query(value))
            except Exception as exc:  # noqa: BLE001 - surface encoder failures to the client
                send_json(500, {"error": str(exc)})
                return

            send_json(
                200,
                {
                    "model": model_name,
                    "dimensions": dimensions,
                    "format": "sparse",
                    "embeddings": embeddings,
                },
            )

    return Handler


def main():
    args = parse_args()
    validate_bind_host(args)
    handler = make_server(args.model, args.dimensions, args.device)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
