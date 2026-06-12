#!/usr/bin/env python3
"""Late-interaction (multi-vector) embedding server for ClawGallery VDR.

Serves vidore/colqwen2-v1.0 via colpali-engine. Speaks the same /embed
contract as jina_omni_server.py, but returns one multi-vector per input:
``embeddings`` is a list of ``[[f32, ...], ...]`` (one 128-dim vector per
image patch / query token) instead of a single pooled vector.

Requires: pip install colpali-engine torch pillow
"""
import argparse
import importlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_MODEL = "vidore/colqwen2-v1.0"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    return parser.parse_args()


def choose_device(requested):
    if requested != "auto":
        return requested
    torch = importlib.import_module("torch")
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def make_server(model_name, device):
    torch = importlib.import_module("torch")
    image_module = importlib.import_module("PIL.Image")
    colpali = importlib.import_module("colpali_engine.models")
    model = colpali.ColQwen2.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        device_map=device,
    ).eval()
    processor = colpali.ColQwen2Processor.from_pretrained(model_name)

    def embed_images(images):
        batch = processor.process_images(images).to(model.device)
        with torch.no_grad():
            return model(**batch)

    def embed_texts(texts):
        batch = processor.process_queries(texts).to(model.device)
        with torch.no_grad():
            return model(**batch)

    def to_multivectors(tensor):
        # tensor: (batch, tokens, dim); drop zero-padded rows.
        multivectors = []
        for doc in tensor.to(torch.float32).cpu():
            rows = [row.tolist() for row in doc if float(row.abs().sum()) > 0.0]
            multivectors.append(rows or [[0.0] * doc.shape[-1]])
        return multivectors

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/embed":
                self.send_error(404, "not found")
                return
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length))
            opened = []
            try:
                images, texts, order = [], [], []
                for item in payload.get("inputs", []):
                    if item.get("kind") == "image":
                        image = image_module.open(Path(item["value"])).convert("RGB")
                        opened.append(image)
                        order.append(("image", len(images)))
                        images.append(image)
                    else:
                        order.append(("text", len(texts)))
                        texts.append(str(item.get("value", "")))
                image_vectors = to_multivectors(embed_images(images)) if images else []
                text_vectors = to_multivectors(embed_texts(texts)) if texts else []
                embeddings = [
                    image_vectors[index] if kind == "image" else text_vectors[index]
                    for kind, index in order
                ]
                dimensions = len(embeddings[0][0]) if embeddings else 0
                body = json.dumps(
                    {
                        "model": model_name,
                        "dimensions": dimensions,
                        "embeddings": embeddings,
                    }
                ).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                body = json.dumps({"error": str(exc)}).encode()
                self.send_response(500)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            finally:
                for image in opened:
                    image.close()

        def log_message(self, format, *args):
            return

    return Handler


def main():
    args = parse_args()
    device = choose_device(args.device)
    handler = make_server(args.model, device)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        json.dumps(
            {
                "url": f"http://{args.host}:{args.port}",
                "model": args.model,
                "device": device,
            }
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
