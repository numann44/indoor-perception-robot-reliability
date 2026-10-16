"""Offline v2 quota/transport tests. Scratch stays under project cache."""
import importlib.util,json,tempfile,time,unittest,urllib.error
from pathlib import Path
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("data_v2_transport",ROOT/"tools/data_v2.py")
transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)


class BudgetTransportTests(unittest.TestCase):
    def test_explicit_v2_quota_and_single_carry_preserve_old_ledger(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"cache/tmp") as temporary:
            root=Path(temporary);old=root/"reports/data/download_ledger.jsonl";old.parent.mkdir(parents=True);original=json.dumps({"cumulative_bytes":1234})+"\n";old.write_text(original)
            first=transport.BudgetV2(root);second=transport.BudgetV2(root)
            self.assertEqual(first.limit,38_000_000_000);self.assertEqual(first.project_limit,40_000_000_000);self.assertEqual(first.min_free,15_000_000_000);self.assertEqual(first.total(),1234);self.assertEqual(second.total(),1234);self.assertEqual(old.read_text(),original);self.assertEqual(len(first.ledger.read_text().splitlines()),1)
    def test_cached_project_oversize_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"cache/tmp") as temporary:
            root=Path(temporary);budget=transport.BudgetV2(root);snapshot=root/"reports/v2/data/storage-snapshot.json";snapshot.write_text(json.dumps({"time_unix":time.time(),"project_bytes":40_000_000_000}))
            with self.assertRaisesRegex(RuntimeError,"40GB"):budget.check_storage()
    def test_503_backoff_reduces_gate_and_response_close_releases_once(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"cache/tmp") as temporary:
            http=transport.HTTPV2(transport.BudgetV2(Path(temporary)));response=Mock();error=urllib.error.HTTPError("https://example.invalid",503,"unavailable",{"Retry-After":"0"},None)
            with patch.object(transport.legacy.HTTP,"response",side_effect=[error,response]),patch.object(transport.time,"sleep") as sleep:
                wrapped=http.response("https://example.invalid",0,9);self.assertTrue(http.throttled);self.assertEqual(http.gate4._value,3);wrapped.close();wrapped.close();self.assertEqual(http.gate4._value,4);response.close.assert_called_once();sleep.assert_called_once_with(1)


if __name__=="__main__":unittest.main()
