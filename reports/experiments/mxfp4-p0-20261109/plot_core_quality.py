"""Plot complete paired core scores, keeping frozen point gates separate from CIs."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
A = HERE.parent / "upstream-production-kv-20261102"
OLD = HERE.parent / "mxfp4-fp8-20261105"
SOURCES = [
    ("FP8 original", A / "quality-pair-A-FP8-a1.json", "#0072B2"),
    ("NVFP4", A / "quality-pair-A-NVFP4-a1.json", "#D55E00"),
    ("FP8 retained", OLD / "quality-pair-A-FP8-retained-a1.json", "#AA6699"),
    ("MXFP4 repaired", HERE / "quality-pair-A-MXFP4-a1.json", "#B22222"),
]

fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.5))
rows, identities = [], {}
for y, (label, path, color) in enumerate(SOURCES):
    raw = path.read_bytes()
    pair = json.loads(raw)
    assert pair["same_frozen_official_base_A_C"] is True
    assert pair["reference_run"] == "upstream-A-quality1720-r0-20261102a"
    assert (pair["bootstrap_seed"], pair["bootstrap_repeats"]) == (20261027, 4000)
    identities[str(path.relative_to(HERE.parent))] = hashlib.sha256(raw).hexdigest()
    for axis, (metric, task) in zip(axes, [("GSM8K", "gsm8k"), ("LongBench Macro", None),
                                         ("RepoBench-P", "repobench-p")]):
        if task is None:
            delta = pair["LongBench_core_macro_delta_pp"]
            low, high = pair["LongBench_macro_paired_95pct_context_cluster_CI_pp"]
        else:
            value = pair["task_differences"][task]
            delta = value["delta_percentage_points"]
            low, high = value["paired_95pct_context_cluster_CI_pp"]
        axis.plot([low, high], [y, y], color=color, lw=2)
        axis.scatter([delta], [y], color=color, s=36, zorder=3)
        rows.append(dict(format=label, metric=metric, delta_pp=delta, CI95_low_pp=low, CI95_high_pp=high))
for axis, title, margin in zip(axes, ["GSM8K flexible EM", "LongBench eight-task Macro", "RepoBench-P"], [.5, 1., 2.]):
    axis.axvline(0, color="#444444", lw=1)
    axis.axvline(-margin, color="#777777", ls="--", lw=1)
    axis.set(title=title, yticks=range(4), yticklabels=[x[0] for x in SOURCES], ylim=(3.5, -.5),
             xlabel="Difference vs BF16 A\n(percentage points)")
    axis.grid(axis="x", alpha=.18)
    axis.spines[["top", "right"]].set_visible(False)
fig.suptitle("Core quality after MXFP4 Graph repair", y=.98, fontsize=13)
fig.text(.5, .03, "95% paired bootstrap intervals. Dashed lines: frozen point-gate limits (-0.5 / -1 / -2 pp).\n"
         "1319 GSM8K + 8 x 50 LongBench; the one extra long document is excluded from Macro.",
         ha="center", fontsize=9, color="#444444")
fig.tight_layout(rect=(0, .13, 1, .94))
for suffix in ("png", "svg"):
    fig.savefig(HERE / ("quality_core." + suffix), dpi=180)
plt.close(fig)
with (HERE / "quality_core_delta_exact.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
with (HERE / "quality_core_plot_provenance.json").open("w") as stream:
    json.dump(dict(source_sha256=identities, frozen_point_margins_pp=[.5, 1., 2.],
                   core_generation_count=1720, tool_results_not_inferred=True,
                   statistical_noninferiority_or_production_acceptance=False), stream, indent=2)
    stream.write("\n")
print(json.dumps({"paired_metrics": len(rows), "figures": ["quality_core.png", "quality_core.svg"]}))
