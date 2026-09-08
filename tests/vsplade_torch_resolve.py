"""Unit tests for Windows V-SPLADE repo-id resolution.

Upstream VSPLADEInference.from_pretrained wraps its argument in Path().
On Windows that turns Hub ids like ``naver/v-splade-efficient`` into
``naver\\v-splade-efficient``, which huggingface_hub rejects. The server must
snapshot_download the repo first and pass a real local directory.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import vsplade_torch_server as server  # noqa: E402


class ResolvePretrainedSourceTests(unittest.TestCase):
    def test_repo_id_uses_downloader_not_path(self) -> None:
        calls: list[dict[str, str]] = []

        def fake_download(*, repo_id: str) -> str:
            calls.append({"repo_id": repo_id})
            self.assertNotIn("\\", repo_id)
            return "/abs/models--naver--v-splade-efficient"

        got = server.resolve_pretrained_source(
            "naver/v-splade-efficient", downloader=fake_download
        )
        self.assertEqual(got, "/abs/models--naver--v-splade-efficient")
        self.assertEqual(calls, [{"repo_id": "naver/v-splade-efficient"}])

    def test_existing_local_dir_skips_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calls: list[dict[str, str]] = []

            def fake_download(*, repo_id: str) -> str:
                calls.append({"repo_id": repo_id})
                return "should-not-run"

            got = server.resolve_pretrained_source(tmp, downloader=fake_download)
            self.assertEqual(Path(got).resolve(), Path(tmp).resolve())
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
