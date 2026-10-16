"""Bounded recorded-publisher / inference-worker / consumer integration.

Offline replay is deterministic and uses sensor timestamps. Realtime replay uses
the wall clock and a real inference thread; scheduler-dependent drops are expected.
"""

from collections import Counter, deque
import threading
import time


def _result_summary(output):
    return {"detections": len(getattr(output, "detections", []) or []),
            "has_semantics": getattr(output, "semantic", None) is not None,
            "has_forecast": getattr(output, "forecast", None) is not None}


def replay_system(frames, adapter, *, mode="offline", worker_delay_s=0.0,
                  queue_capacity=2, max_duration_s=60.0, return_outputs=False):
    """Run replay and return outputs, counters and actual queue occupancy.

    Duplicate sample IDs or timestamps are rejected before enqueue. A timestamp
    <= the latest processed timestamp is rejected by both publisher and worker.
    Every queue mutation is locked; an in-flight frame is separate from capacity.
    """
    if mode not in ("offline", "realtime"):
        raise ValueError("mode must be offline or realtime")
    if queue_capacity < 1 or worker_delay_s < 0 or max_duration_s <= 0:
        raise ValueError("Invalid queue capacity, delay or duration")
    counters = Counter(published=0, processed=0, duplicate=0, out_of_order=0,
                       oldest_dropped=0, timeout_dropped=0, inference_error=0)
    results, errors, outputs = [], [], []
    seen_ids, seen_times = set(), set()
    latest_processed = -1
    max_observed = 0
    worker_alive = False

    def consume(frame, measure_wall, enqueued_ns=None):
        nonlocal latest_processed
        started = time.perf_counter_ns()
        try:
            prediction = adapter.predict(frame)
            if worker_delay_s and measure_wall:
                time.sleep(worker_delay_s)
            elapsed = (time.perf_counter_ns() - started) / 1e6 if measure_wall else None
            queue_wait = (started - enqueued_ns) / 1e6 if enqueued_ns is not None else None
            publish_latency = (time.perf_counter_ns() - enqueued_ns) / 1e6 if enqueued_ns is not None else None
            latest_processed = frame.timestamp_ns
            counters["processed"] += 1
            results.append({"sample_id": frame.sample_id, "timestamp_ns": frame.timestamp_ns,
                            "latency_ms": elapsed, **_result_summary(prediction)})
            results[-1]["queue_wait_ms"] = queue_wait
            results[-1]["publish_to_consumer_ms"] = publish_latency
            if return_outputs:
                outputs.append(prediction)
        except Exception as exc:
            counters["inference_error"] += 1
            errors.append({"sample_id": frame.sample_id, "error": f"{type(exc).__name__}: {exc}"})

    def admission(frame):
        counters["published"] += 1
        if frame.sample_id in seen_ids or frame.timestamp_ns in seen_times:
            counters["duplicate"] += 1
            return False
        if frame.timestamp_ns <= latest_processed:
            counters["out_of_order"] += 1
            return False
        seen_ids.add(frame.sample_id)
        seen_times.add(frame.timestamp_ns)
        return True

    started = time.monotonic()
    if mode == "offline":
        for frame in frames:
            if admission(frame):
                consume(frame, False)
        max_observed = min(1, counters["processed"])
    else:
        queue = deque()
        condition = threading.Condition()
        publisher_done = False
        cancelled = False

        def worker():
            nonlocal latest_processed
            while True:
                with condition:
                    condition.wait_for(lambda: queue or publisher_done or cancelled)
                    if cancelled or (publisher_done and not queue):
                        return
                    frame, enqueued_ns = queue.popleft()
                    if frame.timestamp_ns <= latest_processed:
                        counters["out_of_order"] += 1
                        continue
                consume(frame, True, enqueued_ns)

        thread = threading.Thread(target=worker, name="robotbench-inference", daemon=True)
        thread.start()
        first_timestamp = None
        wall_origin = time.monotonic()
        try:
            for frame in frames:
                if first_timestamp is None:
                    first_timestamp = frame.timestamp_ns
                target = wall_origin + max(0, frame.timestamp_ns - first_timestamp) / 1e9
                remaining = target - time.monotonic()
                if remaining > 0:
                    time.sleep(min(remaining, max(0, started + max_duration_s - time.monotonic())))
                if time.monotonic() - started >= max_duration_s:
                    counters["timeout_dropped"] += 1
                    break
                with condition:
                    if not admission(frame):
                        continue
                    if len(queue) >= queue_capacity:
                        queue.popleft()
                        counters["oldest_dropped"] += 1
                    queue.append((frame, time.perf_counter_ns()))
                    max_observed = max(max_observed, len(queue))
                    condition.notify()
        finally:
            with condition:
                publisher_done = True
                condition.notify_all()
        thread.join(max(0.0, started + max_duration_s - time.monotonic()))
        if thread.is_alive():
            worker_alive = True
            with condition:
                cancelled = True
                counters["timeout_dropped"] += len(queue)
                queue.clear()
                condition.notify_all()
            # A Python thread cannot safely kill an uninterruptible model call.
            # Mark incomplete; daemon lifetime and caller timeout bound the process.
            errors.append({"error": "Inference worker exceeded replay deadline; in-flight prediction may still finish."})
    duration = time.monotonic() - started
    return {"mode": mode, "clock": "sensor" if mode == "offline" else "wall",
            "counters": dict(counters), "queue_capacity": queue_capacity,
            "max_queue_observed": max_observed, "results": list(results),
            "errors": list(errors), "complete": not errors, "worker_alive_at_deadline": worker_alive,
            "duration_s": duration if mode == "realtime" else None,
            **({"outputs": list(outputs)} if return_outputs else {})}
