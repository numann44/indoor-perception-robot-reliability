"""Acquisition evidence survives local re-admission; no network or model work."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("data_v2_admission", ROOT / "tools/data_v2.py")
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)


class AdmissionTests(unittest.TestCase):
    def _scratch(self):
        return tempfile.TemporaryDirectory(dir=ROOT / "cache/tmp")

    def _bag(self, root, payload=b"#ROSBAG V2.0\ncausal sensor bytes"):
        bag = root / "data/v2/openloris/office1-1.bag"
        bag.parent.mkdir(parents=True)
        bag.write_bytes(payload)
        report = root / "reports/v2/data/openloris-download.json"
        report.parent.mkdir(parents=True)
        original = {"member": "office1-1.bag", "path": str(bag.relative_to(root)),
                    "bytes": len(payload), "bag_sha256": hashlib.sha256(payload).hexdigest(),
                    "chunk_proofs": [{"start": 512, "end": 512 + len(payload) - 1,
                                      "sha256": hashlib.sha256(payload).hexdigest()}]}
        report.write_text(json.dumps(original, indent=3) + "\n")
        return bag, report

    def test_two_existing_bag_fetches_preserve_exact_acquisition_and_make_unique_rechecks(self):
        with self._scratch() as temporary:
            root = Path(temporary)
            bag, report = self._bag(root)
            original = report.read_bytes()
            with patch.object(transport, "ROOT", root), patch.object(transport, "OPENLORIS_BAG_BYTES", bag.stat().st_size), patch.object(transport, "http_v2", side_effect=AssertionError("existing-file fetch must be offline")):
                first = transport.fetch_openloris()
                second = transport.fetch_openloris()
            self.assertNotEqual(first, second)
            self.assertEqual(report.read_bytes(), original)
            for recheck in (first, second):
                proof = json.loads(recheck.read_text())
                self.assertEqual(proof["http_requests"], 0)
                self.assertEqual(proof["new_download_bytes"], 0)
                self.assertFalse(proof["historical_report_replaced"])
                self.assertEqual((root / proof["acquisition_report"]["immutable_copy"]).read_bytes(), original)

    def test_corrupt_existing_bag_is_rejected_before_network_or_report_rewrite(self):
        with self._scratch() as temporary:
            root = Path(temporary)
            bag, report = self._bag(root)
            original = report.read_bytes()
            bag.write_bytes(bag.read_bytes()[:-1] + b"X")
            with patch.object(transport, "ROOT", root), patch.object(transport, "OPENLORIS_BAG_BYTES", bag.stat().st_size), patch.object(transport, "http_v2", side_effect=AssertionError("no repair download")):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    transport.fetch_openloris()
            self.assertEqual(report.read_bytes(), original)
            self.assertFalse((root / "reports/v2/data/rechecks").exists())

    def test_missing_bag_report_cannot_be_recreated_as_historical_proof(self):
        with self._scratch() as temporary:
            root = Path(temporary)
            bag, report = self._bag(root)
            report.unlink()
            with patch.object(transport, "ROOT", root), patch.object(transport, "http_v2", side_effect=AssertionError("no network")):
                with self.assertRaisesRegex(ValueError, "lacks acquisition report"):
                    transport.fetch_openloris()
            self.assertFalse(report.exists())

    def test_later_acquisition_record_never_overwrites_first_complete_report(self):
        with self._scratch() as temporary:
            root = Path(temporary)
            with patch.object(transport, "ROOT", root):
                initial = transport._write_acquisition_once("sample", {"chunk_proofs": ["original"]})
                original = initial.read_bytes()
                later = transport._write_acquisition_once("sample", {"chunk_proofs": [], "reused": True})
            self.assertEqual(initial.read_bytes(), original)
            proof = json.loads(later.read_text())
            self.assertEqual(proof["current_execution"]["chunk_proofs"], [])
            self.assertEqual((root / proof["prior_acquisition_report"]["immutable_copy"]).read_bytes(), original)

    def test_frozen_index_change_is_rejected_without_decoder_or_rewrite(self):
        with self._scratch() as temporary:
            root = Path(temporary)
            index = root / "data/v2/jaad/index.jsonl"
            index.parent.mkdir(parents=True)
            index.write_text('{"split":"train"}\n')
            report = root / "reports/v2/data/jaad-validation.json"
            report.parent.mkdir(parents=True)
            report.write_text(json.dumps({"primary_ready": True, "index_sha256": "0" * 64,
                                          "counts": {"train": 1}, "files": []}))
            original = report.read_bytes()
            with patch.object(transport, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    transport._frozen_index_admission("jaad", {})
            self.assertEqual(report.read_bytes(), original)

    def test_artifact_outside_project_is_rejected(self):
        with self._scratch() as temporary:
            root = Path(temporary)
            with patch.object(transport, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "outside project"):
                    transport._verify_artifact("../external.bag")


if __name__ == "__main__":
    unittest.main()
