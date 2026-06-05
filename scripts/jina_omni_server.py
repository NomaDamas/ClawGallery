#!/usr/bin/env python3
import argparse
import importlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default="jinaai/jina-embeddings-v5-omni-small")
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


def normalize(vector):
    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def make_server(model_name, device):
    image_module = importlib.import_module("PIL.Image")
    sentence_transformers = importlib.import_module("sentence_transformers")
    sentence_transformer = sentence_transformers.SentenceTransformer

    model = sentence_transformer(model_name, trust_remote_code=True, device=device)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/embed":
                self.send_error(404, "not found")
                return
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length))
            dimensions = int(payload.get("dimensions") or 1024)
            items = []
            opened = []
            try:
                for item in payload.get("inputs", []):
                    if item.get("kind") == "image":
                        image = image_module.open(Path(item["value"])).convert("RGB")
                        opened.append(image)
                        items.append(image)
                    else:
                        items.append(str(item.get("value", "")))
                encoded = model.encode(
                    items,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                embeddings = []
                for vector in encoded.tolist():
                    embeddings.append(normalize(vector[:dimensions]))
                body = json.dumps(
                    {
                        "model": model_name,
                        "dimensions": len(embeddings[0]) if embeddings else dimensions,
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
