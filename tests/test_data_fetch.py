"""Offline HTTP/ZIP safety and synchronization tests; all scratch files project-local."""
import importlib.util
import gzip
import io
import json
from pathlib import Path
import sys
import tempfile
import tarfile
import threading
import unittest
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("portfolio_data_fetch", ROOT / "tools/data_fetch.py")
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)
spec2 = importlib.util.spec_from_file_location("portfolio_data_validate", ROOT / "tools/data_validate.py")
validate = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(validate)


class RangeHandler(BaseHTTPRequestHandler):
    payload = b""
    mode = "good"
    def log_message(self, *args):
        pass
    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", len(self.payload))
        self.end_headers()
    def do_GET(self):
        header = self.headers.get("Range")
        if self.mode == "ignore" or not header:
            self.send_response(200)
            self.send_header("Content-Length", len(self.payload))
            self.end_headers()
            self.wfile.write(self.payload)
            return
        start, end = map(int, header.removeprefix("bytes=").split("-"))
        self.send_response(206)
        total = len(self.payload) + (1 if self.mode == "bad_total" else 0)
        self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
        self.send_header("Content-Length", end - start + 1)
        self.end_headers()
        body = bytearray(self.payload[start:end+1])
        if self.mode == "corrupt" and len(body) > 30:
            body[-1] ^= 255
        self.wfile.write(body)


class DataFetchTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / "cache/tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="data-fetch-test-", dir=scratch)
        self.path = Path(self.temp.name)
        self.original_root = fetch.ROOT
        fetch.ROOT = self.path
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
            z.writestr("safe/payload.bin", b"test payload" * 100)
        RangeHandler.payload, RangeHandler.mode = buf.getvalue(), "good"
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/archive.zip"
        self.budget = fetch.Budget(self.path, limit=1000000, project_limit=1000000, min_free=0)
        self.http = fetch.HTTP(self.budget)
    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        fetch.ROOT = self.original_root
        self.temp.cleanup()
    def test_range_extract_crc_sha_and_resume(self):
        remote = fetch.RemoteZipFile(self.url, self.http)
        with zipfile.ZipFile(remote) as z:
            info = z.getinfo("safe/payload.bin")
        out = self.path / "data/payload.bin"
        report = fetch.extract_member(remote, info, out)
        self.assertEqual(out.read_bytes(), b"test payload" * 100)
        self.assertEqual(len(report["sha256"]), 64)
        self.assertTrue(fetch.extract_member(remote, info, out)["reused"])
    def test_parallel_range_archive_assembly_is_byte_identical(self):
        destination = self.path / "cache/downloads/source.tgz"
        result = fetch.range_download(self.http, self.url, destination, workers=2, chunk_bytes=128)
        self.assertEqual(destination.read_bytes(), RangeHandler.payload)
        self.assertEqual(result["sha256"], fetch.sha256(destination))
        self.assertGreater(result["range_chunks"], 1)
        self.assertFalse(destination.parent.joinpath(destination.name + ".chunks").exists())
    def test_server_ignoring_range_is_rejected_without_body(self):
        RangeHandler.mode = "ignore"
        with self.assertRaisesRegex(RuntimeError, "Range rejected before body"):
            self.http.read(self.url, start=0, end=9)
        self.assertEqual(self.budget.total(), 0)
    def test_wrong_content_range_total_is_rejected(self):
        self.http.head(self.url)
        RangeHandler.mode = "bad_total"
        with self.assertRaisesRegex(RuntimeError, "Range rejected"):
            self.http.read(self.url, start=0, end=9)
    def test_zip_member_corruption_is_rejected_and_part_removed(self):
        remote = fetch.RemoteZipFile(self.url, self.http)
        with zipfile.ZipFile(remote) as z:
            info = z.getinfo("safe/payload.bin")
        RangeHandler.mode = "corrupt"
        out = self.path / "data/payload.bin"
        with self.assertRaisesRegex(RuntimeError, "CRC"):
            fetch.extract_member(remote, info, out)
        self.assertFalse(out.exists())
        self.assertFalse(out.with_suffix(".bin.part").exists())
    def test_archive_traversal_is_rejected(self):
        for name in ["../escape", "/absolute", "nested/../../escape", "nested\\escape"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                fetch.safe_path(self.path, name)
    def test_download_ledger_counts_every_retry_chunk(self):
        self.budget.record(10, self.url, "test", attempt=1)
        self.budget.record(7, self.url, "test", attempt=2)
        self.assertEqual(self.budget.total(), 17)
        records = [json.loads(x) for x in self.budget.ledger.read_text().splitlines()]
        self.assertEqual([x["attempt"] for x in records], [1, 2])
    def test_quota_preflight_prevents_network_body(self):
        self.budget.limit = 5
        with self.assertRaisesRegex(RuntimeError, "budget exceeded"):
            self.http.read(self.url, start=0, end=9)
        self.assertEqual(self.budget.total(), 0)
    def test_timestamp_matching_is_unique_and_bounded(self):
        rgb = [(1.0, []), (1.01, []), (2.0, [])]
        depth = [(1.005, []), (2.03, [])]
        pairs = validate.associate(rgb, depth, 0.02)
        self.assertEqual(len(pairs), 1)
        self.assertLessEqual(pairs[0][2], 0.02)
    def test_nonmonotonic_sensor_index_is_rejected(self):
        p = self.path / "timestamps.txt"
        p.write_text("1.0 a.png\n1.0 b.png\n")
        with self.assertRaisesRegex(ValueError, "strictly"):
            validate.read_timestamp_file(p)
    def test_conflicting_duplicate_pose_times_are_all_discarded(self):
        p = self.path / "poses.txt"
        p.write_text("0.0 0 0 0 0 0 0 1\n0.1 1 0 0 0 0 0 1\n0.1 2 0 0 0 0 0 1\n0.2 3 0 0 0 0 0 1\n")
        poses, report = validate.read_pose_file(p)
        self.assertEqual([x[0] for x in poses], [0.0, 0.2])
        self.assertEqual(report["discarded_ambiguous_records"], 2)
        self.assertEqual(report["duplicate_timestamps"], [0.1])
    def test_tar_path_escape_and_links_are_rejected(self):
        for name, link in [("../escape.bin", False), ("link", True)]:
            archive = self.path / "unsafe.tgz"
            with tarfile.open(archive, "w:gz") as tar:
                info = tarfile.TarInfo(name)
                if link:
                    info.type, info.linkname = tarfile.SYMTYPE, "../escape"
                    tar.addfile(info)
                else:
                    info.size = 1
                    tar.addfile(info, io.BytesIO(b"x"))
            with self.subTest(name=name), self.assertRaises(ValueError):
                fetch.safe_extract_tar(archive, self.path / "unpacked")
    def test_gzip_footer_corruption_is_rejected(self):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo("data/file.txt")
            info.size = 5
            tar.addfile(info, io.BytesIO(b"hello"))
        body = bytearray(gzip.compress(buf.getvalue()))
        body[-8] ^= 255
        archive = self.path / "corrupt.tgz"
        archive.write_bytes(body)
        # Budget(root=ROOT) captures its default before setUp replaces ROOT.
        # Exercise the real quota checks on this test's isolated scratch root.
        with mock.patch.object(fetch, "Budget", return_value=self.budget), self.assertRaises((gzip.BadGzipFile, OSError)):
            fetch.safe_extract_tar(archive, self.path / "unpacked")
    def test_tonemap_uses_rgb_only_and_official_gamma_formula(self):
        import numpy as np
        image = np.ones((4, 4, 3), dtype=np.float32)
        mapped, parameters = validate.tonemap(image, np)
        self.assertTrue(np.all(mapped == 204))
        self.assertIn("no semantic/instance/render IDs", parameters["validity"])
        image[0, 0] = np.nan
        mapped, _ = validate.tonemap(image, np)
        self.assertTrue(np.all(mapped[0, 0] == 0))
    def test_pose_slerp_handles_sign_equivalent_quaternions(self):
        import numpy as np
        poses = [(0.0, [0,0,0,0,0,0,1]), (0.1, [1,0,0,0,0,0,-1])]
        result = validate.pose_at(0.05, poses, np)
        self.assertAlmostEqual(result[0], 0.5)
        self.assertAlmostEqual(np.linalg.norm(result[3:]), 1.0)
        self.assertIsNone(validate.pose_at(0.5, poses, np))


if __name__ == "__main__":
    unittest.main()
