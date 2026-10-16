"""Static exportable plots generated only from recorded measurements."""

import json
import os
from pathlib import Path


def write_plots(summary_path, output_dir=None):
    summary_path = Path(summary_path).resolve()
    output_dir = Path(output_dir).resolve() if output_dir else summary_path.parent
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    summary = json.loads(summary_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [(name, data) for name, data in summary["conditions"].items()
            if data["adapter_latency"].get("count", 0)]
    labels = [name for name, _ in rows]
    x = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(10, max(3, len(rows) * 0.35)))
    axis.barh(x - 0.18, [data["adapter_latency"]["p50_ms"] for _, data in rows], height=0.35, label="Adapter p50")
    axis.barh(x + 0.18, [data["adapter_latency"]["p95_ms"] for _, data in rows], height=0.35, label="Adapter p95")
    axis.set_yticks(x, labels); axis.invert_yaxis()
    axis.set_xlabel("Latency (ms), preprocessing + synchronized inference + postprocessing")
    axis.set_title(f"{summary['config']['backend']} / {summary['config']['device']} · batch 1 · {summary['config']['threads']} requested threads")
    axis.legend(); axis.grid(axis="x", alpha=0.2)
    figure.text(0.02, 0.01, f"Source: {summary_path.name}; run {summary['run_id']} · actual subprocess repeats {len(summary['repeat_evidence'])}", fontsize=8)
    figure.tight_layout(rect=(0, 0.025, 1, 1))
    latency = output_dir / "adapter-latency.png"
    figure.savefig(latency, dpi=160); plt.close(figure)
    figure, axis = plt.subplots(figsize=(6, 3.2))
    runs = summary["repeat_evidence"]
    axis.bar([str(row["repeat"]) for row in runs], [row["memory_pass"]["rss_peak_observed_bytes"] / 1024**2 for row in runs])
    axis.set_xlabel("Independent subprocess repeat"); axis.set_ylabel("Process RSS (MiB)")
    axis.set_title("Separate memory pass · 2 ms sampled process peak")
    figure.text(0.02, 0.01, "Excludes accelerator allocation; a sampled peak can miss brief allocations.", fontsize=8)
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    memory = output_dir / "process-memory.png"
    figure.savefig(memory, dpi=160); plt.close(figure)
    evidence = {"source_summary": str(summary_path), "run_id": summary["run_id"],
                "images": [str(latency), str(memory)]}
    (output_dir / "plot-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return evidence
