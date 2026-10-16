#!/usr/bin/env python3
"""Anonymous, bounded official-source acquisition. Never execute downloaded code."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import fcntl
import gzip
import hashlib
import io
import json
import os
import re
from pathlib import Path, PurePosixPath
import shutil
import struct
import tarfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_LIMIT = 7_000_000_000  # One additional GB reserved for other project dependencies.
PROJECT_LIMIT = 8_000_000_000
MIN_FREE = 15_000_000_000
UA = "IndoorPerceptionPortfolio/0.1 (local academic research)"


def safe_path(root: Path, member: str) -> Path:
    p = PurePosixPath(member)
    if p.is_absolute() or ".." in p.parts or "\\" in member:
        raise ValueError(f"Unsafe archive member: {member}")
    out = (root / member).resolve()
    if not out.is_relative_to(root.resolve()):
        raise ValueError(f"Archive path escapes root: {member}")
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


class Budget:
    """Append-only byte ledger, also safe when TUM and ZIP acquisition overlap."""
    def __init__(self, root=ROOT, limit=DOWNLOAD_LIMIT, project_limit=PROJECT_LIMIT,
                 min_free=MIN_FREE):
        self.root = Path(root).resolve()
        self.limit, self.project_limit, self.min_free = limit, project_limit, min_free
        self.ledger = self.root / "reports/data/download_ledger.jsonl"
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._last_disk_check = 0.0

    def check_storage(self, force=False):
        now = time.monotonic()
        if not force and now - self._last_disk_check < 10:
            return
        free = shutil.disk_usage(self.root).free
        size = 0
        for p in self.root.rglob("*"):
            try:
                if p.is_file() and not p.is_symlink():
                    size += p.stat().st_size
            except FileNotFoundError:
                pass  # Another project worker atomically finalized its own cache file.
        if free < self.min_free or size >= self.project_limit:
            raise RuntimeError(f"Storage quota: project={size}, free={free}")
        self._last_disk_check = now

    def _last_total(self, f):
        f.seek(0, os.SEEK_END)
        end = f.tell()
        if not end:
            return 0
        f.seek(max(0, end - 4096))
        lines = f.read().splitlines()
        return json.loads(lines[-1])["cumulative_bytes"]

    def total(self):
        with self._lock, self.ledger.open("a+", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            return self._last_total(f)

    def preflight(self, expected_bytes):
        self.check_storage()
        if self.total() + expected_bytes > self.limit:
            raise RuntimeError("Cumulative download budget exceeded (dependency reserve preserved)")

    def record(self, n, url, purpose, attempt=1):
        with self._lock, self.ledger.open("a+", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            total = self._last_total(f) + n
            event = {"time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                     "bytes": n, "cumulative_bytes": total, "url": url,
                     "purpose": purpose, "attempt": attempt}
            f.seek(0, os.SEEK_END)
            f.write(json.dumps(event, separators=(",", ":")) + "\n")
            f.flush()
        if total > self.limit:
            raise RuntimeError("Download byte quota reached")
        self.check_storage()


class HTTP:
    def __init__(self, budget: Budget):
        self.budget = budget
        self.resource_sizes = {}

    def head(self, url):
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=30) as r:
            if r.status != 200:
                raise RuntimeError(f"HEAD returned {r.status}: {url}")
            self.resource_sizes[url] = int(r.headers["Content-Length"])
            return {**dict(r.headers), "_resolved_url": r.geturl()}

    def response(self, url, start=None, end=None):
        headers = {"User-Agent": UA, "Accept-Encoding": "identity"}
        if start is not None:
            headers["Range"] = f"bytes={start}-{end}"
            self.budget.preflight(end - start + 1)
        req = urllib.request.Request(url, headers=headers)
        r = urllib.request.urlopen(req, timeout=45)
        if start is not None:
            content_range = r.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
            valid = (match is not None and int(match[1]) == start and int(match[2]) == end
                     and int(match[3]) > end
                     and (url not in self.resource_sizes or int(match[3]) == self.resource_sizes[url]))
            if r.status != 206 or not valid:
                r.close()
                raise RuntimeError(f"Range rejected before body read: status={r.status}, range={content_range}")
            if int(r.headers.get("Content-Length", -1)) != end - start + 1:
                r.close()
                raise RuntimeError("Range Content-Length mismatch")
        elif r.status != 200:
            r.close()
            raise RuntimeError(f"GET returned {r.status}: {url}")
        return r

    def read(self, url, max_bytes=8_000_000, start=None, end=None, purpose="metadata"):
        expected = end - start + 1 if start is not None else None
        if expected is not None and expected > max_bytes:
            raise RuntimeError("Metadata request exceeds cap")
        with self.response(url, start, end) as r:
            length = r.headers.get("Content-Length")
            if length and int(length) > max_bytes:
                raise RuntimeError("Response exceeds metadata cap")
            chunks = []
            n = 0
            while True:
                b = r.read(min(1024 * 1024, max_bytes - n + 1))
                if not b:
                    break
                self.budget.record(len(b), url, purpose)
                n += len(b)
                if n > max_bytes:
                    raise RuntimeError("Response exceeded metadata cap")
                chunks.append(b)
        if expected is not None and n != expected:
            raise RuntimeError("Truncated Range response")
        return b"".join(chunks)

    def download(self, url, destination: Path, max_bytes=1_000_000_000, attempts=3):
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            return {"path": str(destination.relative_to(ROOT)), "bytes": destination.stat().st_size,
                    "sha256": sha256(destination), "reused": True}
        last_error = None
        for attempt in range(1, attempts + 1):
            part = destination.with_suffix(destination.suffix + ".part")
            try:
                with self.response(url) as r, part.open("wb") as f:
                    length = int(r.headers.get("Content-Length", 0))
                    if length > max_bytes:
                        raise RuntimeError("Full download exceeds per-file cap")
                    self.budget.preflight(length or max_bytes)
                    n = 0
                    while True:
                        b = r.read(1024 * 1024)
                        if not b:
                            break
                        self.budget.record(len(b), url, "full_download", attempt)
                        n += len(b)
                        if n > max_bytes:
                            raise RuntimeError("Full download exceeded file cap")
                        f.write(b)
                    if length and n != length:
                        raise RuntimeError("Truncated full response")
                part.replace(destination)
                return {"path": str(destination.relative_to(ROOT)), "bytes": n,
                        "sha256": sha256(destination), "url": url}
            except Exception as e:
                last_error = e
                if part.exists() and part.resolve().is_relative_to(ROOT):
                    part.unlink()
                if isinstance(e, (ValueError, RuntimeError)):
                    raise
                print(f"Retry {attempt}: {url}: {e}", flush=True)
        raise RuntimeError(f"Download failed: {url}: {last_error}")


class RemoteZipFile:
    """Read only ZIP metadata using range requests; payload extraction is streamed."""
    def __init__(self, url, http: HTTP):
        self.url, self.http, self.offset = url, http, 0
        headers = http.head(url)
        self.size = int(headers["Content-Length"])
        self.headers = headers

    def seekable(self):
        return True

    def tell(self):
        return self.offset

    def seek(self, offset, whence=0):
        self.offset = offset if whence == 0 else self.offset + offset if whence == 1 else self.size + offset
        return self.offset

    def read(self, n=-1):
        n = self.size - self.offset if n is None or n < 0 else min(n, self.size - self.offset)
        if n <= 0:
            return b""
        b = self.http.read(self.url, max_bytes=2_000_000, start=self.offset,
                           end=self.offset + n - 1, purpose="zip_metadata")
        self.offset += len(b)
        return b


def extract_member(remote: RemoteZipFile, info: zipfile.ZipInfo, destination: Path):
    if info.flag_bits & 1 or info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
        raise ValueError("Unsupported/encrypted ZIP member")
    safe_path(ROOT, info.filename)  # Check member even when mapped into a shorter local path.
    if not destination.resolve().is_relative_to(ROOT):
        raise ValueError("Output path outside project")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        crc = 0
        with destination.open("rb") as f:
            for b in iter(lambda: f.read(1024 * 1024), b""):
                crc = zlib.crc32(b, crc)
        if destination.stat().st_size == info.file_size and crc & 0xFFFFFFFF == info.CRC:
            return {"member": info.filename, "path": str(destination.relative_to(ROOT)),
                    "bytes": info.file_size, "crc32": f"{info.CRC:08x}",
                    "sha256": sha256(destination), "reused": True}
        raise RuntimeError(f"Existing unverified data requires investigation: {destination}")
    raw = remote.http.read(remote.url, max_bytes=30, start=info.header_offset,
                           end=info.header_offset + 29, purpose="zip_local_header")
    header = struct.unpack("<4s5H3L2H", raw)
    if header[0] != b"PK\x03\x04":
        raise RuntimeError("Invalid local ZIP header")
    if header[3] != info.compress_type or header[2] & 1:
        raise RuntimeError("Local/central ZIP compression flags disagree")
    filename_bytes = remote.http.read(remote.url, max_bytes=65535,
        start=info.header_offset + 30, end=info.header_offset + 29 + header[-2],
        purpose="zip_local_filename")
    encoding = "utf-8" if header[2] & 0x800 else "cp437"
    if filename_bytes.decode(encoding) != info.filename:
        raise RuntimeError("Local/central ZIP filename mismatch")
    start = info.header_offset + 30 + header[-2] + header[-1]
    if info.compress_size == 0:
        raise ValueError("Empty payload member not expected")
    part = destination.with_suffix(destination.suffix + ".part")
    crc, size, digest = 0, 0, hashlib.sha256()
    decompressor = zlib.decompressobj(-15) if info.compress_type == zipfile.ZIP_DEFLATED else None
    try:
        received = 0
        with remote.http.response(remote.url, start, start + info.compress_size - 1) as r, part.open("wb") as f:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                received += len(chunk)
                remote.http.budget.record(len(chunk), remote.url, "zip_member:" + info.filename)
                out = decompressor.decompress(chunk) if decompressor else chunk
                size += len(out)
                if size > info.file_size:
                    raise RuntimeError("ZIP decompression exceeded declared size")
                crc = zlib.crc32(out, crc)
                digest.update(out)
                f.write(out)
            if decompressor:
                out = decompressor.flush()
                size += len(out)
                crc = zlib.crc32(out, crc)
                digest.update(out)
                f.write(out)
                if not decompressor.eof:
                    raise RuntimeError("Incomplete ZIP deflate stream")
        if received != info.compress_size or size != info.file_size or crc & 0xFFFFFFFF != info.CRC:
            raise RuntimeError("ZIP member size/CRC verification failed")
        part.replace(destination)
        return {"member": info.filename, "path": str(destination.relative_to(ROOT)),
                "bytes": size, "compressed_bytes": received, "crc32": f"{info.CRC:08x}",
                "sha256": digest.hexdigest(), "url": remote.url}
    except Exception:
        if part.exists() and part.resolve().is_relative_to(ROOT):
            part.unlink()
        raise


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def range_download(http, url, destination, workers=4, chunk_bytes=4_000_000):
    """Bounded parallel ranges; resume verified chunks, account interrupted attempts."""
    if destination.exists():
        return {"path": str(destination.relative_to(ROOT)), "bytes": destination.stat().st_size,
                "sha256": sha256(destination), "reused": True, "url": url}
    headers = http.head(url)
    size = int(headers["Content-Length"])
    http.budget.preflight(size)
    chunk_dir = destination.parent / (destination.name + ".chunks")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(i, start, min(start + chunk_bytes, size) - 1)
            for i, start in enumerate(range(0, size, chunk_bytes))]
    def get_chunk(job):
        i, start, end = job
        out = chunk_dir / f"{i:06d}.bin"
        proof = chunk_dir / f"{i:06d}.json"
        if out.exists() and proof.exists():
            record = json.loads(proof.read_text())
            if out.stat().st_size == end-start+1 and sha256(out) == record["sha256"]:
                return record
            raise RuntimeError("Existing range chunk failed integrity verification")
        for attempt in range(1, 4):
            part = out.with_suffix(".bin.part")
            try:
                received = 0
                with http.response(url, start, end) as r, part.open("wb") as f:
                    while True:
                        b = r.read(1024 * 1024)
                        if not b:
                            break
                        http.budget.record(len(b), url, f"archive_range:{start}-{end}", attempt)
                        received += len(b)
                        f.write(b)
                if received != end-start+1:
                    raise RuntimeError("Truncated archive range")
                part.replace(out)
                record = {"start": start, "end": end, "bytes": received, "sha256": sha256(out)}
                write_json(proof, record)
                print(f"TUM range ready {destination.name} {i+1}/{len(jobs)}", flush=True)
                return record
            except (OSError, urllib.error.URLError) as e:
                if part.exists():
                    part.unlink()
                if attempt == 3:
                    raise
                print(f"Range retry {attempt} {i}: {e}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(get_chunk, jobs))
    part = destination.with_suffix(destination.suffix + ".part")
    with part.open("wb") as dst:
        for i, _, _ in jobs:
            with (chunk_dir / f"{i:06d}.bin").open("rb") as src:
                shutil.copyfileobj(src, dst, 1024 * 1024)
    if part.stat().st_size != size:
        raise RuntimeError("Assembled archive size mismatch")
    part.replace(destination)
    report = {"path": str(destination.relative_to(ROOT)), "bytes": size, "sha256": sha256(destination),
              "url": url, "range_chunks": len(jobs), "range_workers": workers, "chunk_proofs": records}
    report["resolved_url"] = headers["_resolved_url"]
    for path in chunk_dir.iterdir():
        if path.is_file() and path.resolve().is_relative_to(ROOT):
            path.unlink()
    chunk_dir.rmdir()
    return report


def manifest():
    return json.loads((ROOT / "configs/data_manifest.json").read_text())


def fetch_jaad(http, config):
    url = config["archive_url"]
    remote = RemoteZipFile(url, http)
    with zipfile.ZipFile(remote) as z:
        index = {x.filename: x for x in z.infolist()}
    report = []
    base = f'https://raw.githubusercontent.com/ykotseruba/JAAD/{config["commit"]}/'
    phases = [("train", config["splits"]["train"][:3]), ("val", config["splits"]["val"][:2]),
              ("train", config["splits"]["train"][3:]), ("val", config["splits"]["val"][2:]),
              ("test", config["splits"]["test"])]
    for split, clips in phases:
        split_url = base + f"split_ids/all_videos/{split}.txt"
        raw = http.read(split_url)
        split_file = ROOT / f"data/jaad/official_splits/{split}.txt"
        split_file.parent.mkdir(parents=True, exist_ok=True)
        split_file.write_bytes(raw)
        allowed = set(raw.decode().splitlines())
        def clip_job(clip):
            if clip not in allowed:
                raise ValueError(f"Clip violates official split: {clip}")
            info = index[f"JAAD_clips/{clip}.mp4"]
            item = extract_member(remote, info, ROOT / f"data/jaad/clips/{clip}.mp4")
            item["split"] = split
            annotation_url = base + f"annotations/{clip}.xml"
            annotation = ROOT / f"data/jaad/annotations/{clip}.xml"
            annotation.parent.mkdir(parents=True, exist_ok=True)
            if not annotation.exists():
                annotation.write_bytes(http.read(annotation_url, max_bytes=5_000_000))
            print(f"JAAD {split} {clip} verified", flush=True)
            write_json(ROOT / f"reports/data/jaad_{clip}_download.json", item)
            return item
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            report.extend(pool.map(clip_job, clips))
        write_json(ROOT / "reports/data/jaad_download.json", report)
        print(f"JAAD {split} payload acquisition complete", flush=True)


def fetch_hypersim(http, config):
    base = f'https://raw.githubusercontent.com/apple-aiml-research/ml-hypersim/{config["commit"]}/'
    for source, local in [
        ("evermotion_dataset/analysis/metadata_images_split_scene_v1.csv", "official_split.csv"),
        ("contrib/mikeroberts3000/metadata_camera_parameters.csv", "camera_parameters.csv"),
        ("README.md", "source_README.md"),
        ("code/python/tools/scene_generate_images_tonemap.py", "source_tonemap.py.txt"),
        ("code/cpp/tools/scene_annotation_tool/semantic_label_descs.csv", "semantic_label_descs.csv"),
    ]:
        path = ROOT / "data/hypersim" / local
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(http.read(base + source))
    rows = list(csv.DictReader((ROOT / "data/hypersim/official_split.csv").open()))
    public = {(x["scene_name"], x["camera_name"], int(x["frame_id"])): x["split_partition_name"]
              for x in rows if x["included_in_public_release"] == "True"}
    jobs = [(split, scene) for split, scenes in config["splits"].items() for scene in scenes]
    def scene_job(job):
        split, scene = job
        url = config["archive_pattern"].format(scene=scene)
        remote = RemoteZipFile(url, http)
        with zipfile.ZipFile(remote) as z:
            index = {x.filename: x for x in z.infolist()}
        files = []
        frames = config["frame_exceptions"].get(scene, config["frames_default"])
        for frame in frames:
            if public.get((scene, "cam_00", frame)) != split:
                raise ValueError("Hypersim public release/split mismatch")
            for kind, modality in [("final", "color"), ("geometry", "semantic"), ("geometry", "depth_meters")]:
                name = f"{scene}/images/scene_cam_00_{kind}_hdf5/frame.{frame:04d}.{modality}.hdf5"
                files.append(extract_member(remote, index[name], ROOT / "data/hypersim/raw" / name))
        for name in ["_detail/cam_00/camera_keyframe_positions.hdf5", "_detail/cam_00/camera_keyframe_orientations.hdf5", "_detail/cam_00/camera_keyframe_frame_indices.hdf5", "_detail/cam_00/metadata_camera.csv", "_detail/metadata_scene.csv"]:
            member = scene + "/" + name
            files.append(extract_member(remote, index[member], ROOT / "data/hypersim/raw" / member))
        write_json(ROOT / f"reports/data/hypersim_{scene}_download.json", {"split": split, "scene": scene, "files": files})
        print(f"Hypersim {split} {scene} verified", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(scene_job, jobs))


def safe_extract_tar(archive, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    expanded = 0
    budget = Budget()
    with gzip.open(archive, "rb") as stream:
        with tarfile.open(fileobj=stream, mode="r|") as tar:
            for member in tar:
                if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
                    raise ValueError("TAR links/special files prohibited")
                expanded += member.size
                if expanded > 2_000_000_000:
                    raise RuntimeError("TUM expanded archive exceeds2GB source quota")
                destination = safe_path(out_dir, member.name)
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as src, destination.open("wb") as dst:
                        shutil.copyfileobj(src, dst, 1024 * 1024)
                budget.check_storage()
        # Consume the remainder so gzip footer CRC/length is verified as well.
        while stream.read(1024 * 1024):
            pass


def fetch_tum(http, config):
    report = []
    for sequence, url in config["archives"].items():
        archive = ROOT / "cache/downloads" / (sequence + ".tgz")
        item = range_download(http, url, archive)
        report.append(item)
        out = ROOT / "data/tum/raw"
        marker = ROOT / f"data/tum/{sequence}.extraction_complete"
        if not marker.exists():
            safe_extract_tar(archive, out)
            marker.write_text(item["sha256"] + "\n")
        write_json(ROOT / "reports/data/tum_download.json", report)
        http.budget.check_storage(force=True)
        print(f"TUM {sequence} verified/extracted", flush=True)


def snapshot_sources(http, config):
    sources = config["terms_sources"]
    records = []
    for name, url in sources.items():
        path = ROOT / "data/source_terms" / (name + ".txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(http.read(url, max_bytes=3_000_000, purpose="license_source_snapshot"))
        records.append({"name": name, "url": url, "path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size})
    write_json(ROOT / "reports/data/source_terms.json", records)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", choices=["jaad", "hypersim", "tum", "terms"], default=["jaad", "hypersim", "tum", "terms"])
    args = p.parse_args()
    config = manifest()
    budget = Budget()
    http = HTTP(budget)
    budget.check_storage(force=True)
    for name in args.datasets:
        {"jaad": fetch_jaad, "hypersim": fetch_hypersim, "tum": fetch_tum, "terms": snapshot_sources}[name](http, config[name] if name != "terms" else config)
    print(json.dumps({"status": "acquisition_complete", "download_bytes": budget.total()}), flush=True)


if __name__ == "__main__":
    main()
