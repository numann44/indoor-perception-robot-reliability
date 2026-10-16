from dataclasses import replace
import time
import numpy as np
import pytest
from robot_model_sdk import SensorFrameV1, ModelOutput
from robot_reliability_bench.replay import replay_system
from robot_reliability_bench.corruptions import CONDITIONS, corrupt_frame


class Adapter:
    def predict(self, frame):
        return ModelOutput()


def frames(n=20, fps=200):
    return [SensorFrameV1(sample_id=str(i), timestamp_ns=int(i / fps * 1e9),
                          rgb=np.full((16, 16, 3), 100, dtype=np.uint8),
                          depth=np.ones((16, 16), dtype=np.float32)) for i in range(n)]


def test_offline_deterministic_rejects_duplicate_and_stale():
    sample = frames(4)
    sequence = [sample[0], sample[1], sample[1], replace(sample[2], timestamp_ns=1), sample[2], sample[3]]
    result = replay_system(sequence, Adapter())
    assert result == replay_system(sequence, Adapter())
    assert result["counters"]["duplicate"] == 1
    assert result["counters"]["out_of_order"] == 1
    assert result["counters"]["processed"] == 4


def test_real_worker_causes_oldest_drop_with_bounded_queue():
    result = replay_system(frames(30), Adapter(), mode="realtime", worker_delay_s=0.025)
    assert result["max_queue_observed"] == 2
    assert result["counters"]["oldest_dropped"] > 0
    assert result["counters"]["processed"] > 0
    assert result["counters"]["published"] == 30
    assert result["counters"]["processed"] + result["counters"]["oldest_dropped"] == 30
    assert all(row["queue_wait_ms"] >= 0 for row in result["results"])
    assert all(row["publish_to_consumer_ms"] >= row["latency_ms"] for row in result["results"])


def test_same_timestamp_with_different_id_is_duplicate():
    sample = frames(2)
    result = replay_system([sample[0], replace(sample[1], timestamp_ns=0)], Adapter())
    assert result["counters"]["duplicate"] == 1


def test_missing_depth_and_exception_are_recorded():
    class DepthAdapter:
        def predict(self, frame):
            assert frame.depth is None
            return ModelOutput()
    assert replay_system([replace(f, depth=None) for f in frames(3)], DepthAdapter())["counters"]["processed"] == 3
    class ErrorAdapter:
        def predict(self, frame):
            raise ValueError("Controlled corrupt sensor")
    result = replay_system(frames(3), ErrorAdapter())
    assert result["counters"]["inference_error"] == 3
    assert not result["complete"]


def test_seeded_corruptions_do_not_mutate_inputs():
    original = frames(1)[0]
    rgb, depth = original.rgb.copy(), original.depth.copy()
    for condition in CONDITIONS:
        a = corrupt_frame(original, condition, np.random.default_rng(42))
        b = corrupt_frame(original, condition, np.random.default_rng(42))
        assert (a is None) == (b is None)
        if a is not None:
            np.testing.assert_array_equal(a.rgb, b.rgb)
            assert a.timestamp_ns == b.timestamp_ns
        np.testing.assert_array_equal(original.rgb, rgb)
        np.testing.assert_array_equal(original.depth, depth)
    with pytest.raises(ValueError):
        corrupt_frame(original, "blur:4", np.random.default_rng(42))


def test_bad_replay_config_rejected():
    with pytest.raises(ValueError):
        replay_system([], Adapter(), queue_capacity=0)


def test_inflight_deadline_is_reported_and_returned_snapshot_is_stable():
    class BlockedAdapter:
        def predict(self, frame):
            time.sleep(0.1)
            return ModelOutput()
    result = replay_system(frames(5, fps=1000), BlockedAdapter(), mode="realtime", max_duration_s=0.02)
    assert not result["complete"]
    assert result["worker_alive_at_deadline"]
    assert result["duration_s"] < 0.09
    snapshot = repr(result)
    time.sleep(0.12)
    assert repr(result) == snapshot
