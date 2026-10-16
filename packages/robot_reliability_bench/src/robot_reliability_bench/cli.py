"""robotbench CLI."""

import argparse
from dataclasses import replace
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Independent robot ML measurement and bounded replay")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--model-pack", default="fixture")
    run.add_argument("--backend", choices=("native", "onnx"), default="native")
    run.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    run.add_argument("--samples", type=int, default=100)
    run.add_argument("--warmup", type=int, default=10)
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--threads", type=int, default=4)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--input-manifest")
    run.add_argument("--split", choices=("train", "val", "validation", "test"))
    run.add_argument("--conditions", default="all", help="all or comma-separated clean,jpeg:1,...")
    run.add_argument("--timeout-seconds", type=int, default=1200)
    run.add_argument("--output", required=True)
    worker = sub.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--config", required=True)
    worker.add_argument("--output", required=True)
    replay = sub.add_parser("replay-test", help="Actual deterministic and bounded wall-clock fixture replay")
    replay.add_argument("--output", required=True)
    evaluate = sub.add_parser("evaluate", help="Evaluate saved predictions against separate target files")
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--targets", required=True)
    evaluate.add_argument("--task", required=True, choices=("detection", "semantic", "tracking", "forecast"))
    evaluate.add_argument("--unit", choices=("m", "px", "normalized_image"))
    evaluate.add_argument("--step-seconds", type=float, default=0.1)
    evaluate.add_argument("--num-classes", type=int, default=40)
    evaluate.add_argument("--output", required=True)
    plot = sub.add_parser("plot", help="Render static plots from an actual summary.json")
    plot.add_argument("--summary", required=True)
    plot.add_argument("--output")
    model_replay = sub.add_parser("replay-model", help="Run a real model pack through publisher/queue/worker/consumer")
    model_replay.add_argument("--model-pack", required=True)
    model_replay.add_argument("--backend", choices=("native", "onnx"), default="native")
    model_replay.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    model_replay.add_argument("--mode", choices=("offline", "realtime"), default="realtime")
    model_replay.add_argument("--samples", type=int, default=40)
    model_replay.add_argument("--fixture-fps", type=float, default=100)
    model_replay.add_argument("--input-manifest")
    model_replay.add_argument("--split", choices=("train", "val", "validation", "test"))
    model_replay.add_argument("--worker-delay-ms", type=float, default=0)
    model_replay.add_argument("--max-duration-seconds", type=float, default=60)
    model_replay.add_argument("--output", required=True)
    evaluate_pack = sub.add_parser("evaluate-model", help="Evaluate fixed model-pack accuracy under preset sensor corruptions")
    evaluate_pack.add_argument("--model-pack", required=True)
    evaluate_pack.add_argument("--input-manifest", required=True)
    evaluate_pack.add_argument("--task", choices=("detection", "semantic"), default="detection")
    evaluate_pack.add_argument("--backend", choices=("native", "onnx"), default="native")
    evaluate_pack.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    evaluate_pack.add_argument("--split", choices=("train", "val", "validation", "test"), default="test")
    evaluate_pack.add_argument("--seed", type=int, default=42)
    evaluate_pack.add_argument("--threads", type=int, default=2)
    evaluate_pack.add_argument("--conditions", default="all")
    evaluate_pack.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "_worker":
        from .runner import measure_worker
        result = measure_worker(json.loads(Path(args.config).read_text()))
        Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"status": "verified", "repeat": result["repeat"], "pid": result["pid"]}))
        return 0
    if args.command == "run":
        from .runner import run_benchmark
        result = run_benchmark(args.model_pack, args.output, backend=args.backend, device=args.device,
            samples=args.samples, warmup=args.warmup, repeats=args.repeats, threads=args.threads,
            seed=args.seed, input_manifest=args.input_manifest,
            split=args.split,
            conditions=None if args.conditions == "all" else args.conditions.split(","),
            timeout_s=args.timeout_seconds)
        print(json.dumps({"status": result["status"], "run_id": result["run_id"], "output": result["output_directory"]}))
        return 0 if result["status"] == "verified" else 1
    if args.command == "evaluate":
        from .evaluation import evaluate_prediction_files
        result = evaluate_prediction_files(args.predictions, args.targets, task=args.task,
            output_path=args.output, unit=args.unit, step_seconds=args.step_seconds, num_classes=args.num_classes)
        print(json.dumps({"status": result["metrics"]["status"], "task": args.task, "output": str(Path(args.output).resolve())}))
        return 0
    if args.command == "plot":
        from .plots import write_plots
        print(json.dumps(write_plots(args.summary, args.output)))
        return 0
    if args.command == "replay-model":
        from .integration import replay_model
        result = replay_model(args.model_pack, args.output, backend=args.backend, device=args.device, mode=args.mode,
            samples=args.samples, fixture_fps=args.fixture_fps, input_manifest=args.input_manifest, split=args.split,
            worker_delay_s=args.worker_delay_ms / 1000, max_duration_s=args.max_duration_seconds)
        print(json.dumps({"status": result["status"], "run_id": result["run_id"],
                          "counters": result["result"]["counters"], "output": str(Path(args.output).resolve())}))
        return 0 if result["status"] == "verified" else 1
    if args.command == "evaluate-model":
        from .model_evaluation import evaluate_model
        result = evaluate_model(args.model_pack, args.input_manifest, args.output, task=args.task,
            backend=args.backend, device=args.device, split=args.split, seed=args.seed, threads=args.threads,
            conditions=None if args.conditions == "all" else args.conditions.split(","))
        print(json.dumps({"status": result["status"], "run_id": result["run_id"], "images": result["images"],
                          "task": result["task"], "output": str(Path(args.output).resolve())}))
        return 0
    if args.command == "replay-test":
        from robot_model_sdk import load_adapter, fixture_frames
        from .replay import replay_system
        adapter = load_adapter("fixture")
        frames = list(fixture_frames(n=40, fps=100))
        try:
            invalid = [frames[0], frames[1], frames[1], replace(frames[2], timestamp_ns=1), *frames[2:]]
            offline = replay_system(invalid, adapter, mode="offline")
            offline_repeat = replay_system(invalid, adapter, mode="offline")
            stressed = replay_system(frames, adapter, mode="realtime", worker_delay_s=0.04)
            missing_depth = replay_system([replace(frame, depth=None) for frame in frames], adapter)
            class FailingAdapter:
                def predict(self, frame):
                    raise RuntimeError("Controlled inference exception")
            errors = replay_system(frames[:2], FailingAdapter())
            checks = {"offline_deterministic": offline == offline_repeat,
                      "duplicate_counted": offline["counters"]["duplicate"] == 1,
                      "out_of_order_counted": offline["counters"]["out_of_order"] == 1,
                      "bounded_queue": stressed["max_queue_observed"] <= 2,
                      "overflow_observed": stressed["counters"]["oldest_dropped"] > 0,
                      "rgb_with_missing_depth": missing_depth["counters"]["processed"] == len(frames),
                      "inference_errors_counted": errors["counters"]["inference_error"] == 2}
            result = {"status": "verified" if all(checks.values()) else "failed", "checks": checks,
                      "offline": offline, "realtime_stress": stressed,
                      "missing_depth": missing_depth, "controlled_inference_error": errors}
        finally:
            adapter.close()
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"status": result["status"], "checks": checks, "output": str(output.resolve())}))
        return 0 if result["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
