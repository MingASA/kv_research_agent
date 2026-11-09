"""Export completed, audited scores and service metrics without rounding.

Run with Python 3.12 from any directory after all three new audits exist.
This reads reports only; it does not launch inference or change scoring.
"""

import csv
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
A = HERE.parent / "upstream-production-kv-20261102"
OLD = HERE.parent / "mxfp4-fp8-20261105"
FORMATS = [
    ("BF16 A", A / "A-quality-a1", A / "A-tools-b1/actual_model_audit.json", A / "A-service-c1.json"),
    ("FP8 original", A / "C-fp8-quality-a1", A / "C-fp8-tools-c1/actual_model_audit.json", A / "C-fp8-service-a1.json"),
    ("NVFP4", A / "C-nvfp4-quality-a1", A / "C-nvfp4-tools-b1/actual_model_audit.json", A / "C-nvfp4-service-a1.json"),
    ("FP8 retained", OLD / "FP8-retained-quality-a1", OLD / "FP8-retained-tools-a1/summary.json", OLD / "FP8-retained-service-a1.json"),
    ("MXFP4 historical Graph-confounded", OLD / "MXFP4-quality-a1", OLD / "MXFP4-tools-a1/summary.json", OLD / "MXFP4-service-a1.json"),
    ("MXFP4 repaired", HERE / "MXFP4-quality-a1", HERE / "MXFP4-tools-a1/actual_model_audit.json", HERE / "MXFP4-service-a1.json"),
]
CASES = ("sharegpt-c32", "sharegpt-c64", "govreport-c32")
SOURCES = {}


def read(path):
    raw = path.read_bytes()
    SOURCES[str(path.relative_to(HERE.parent))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def csv_new(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    quality, per_task, services, curves = [], [], [], []
    common = None
    for label, core_path, tool_path, service_path in FORMATS:
        core = read(core_path / "summary.json")
        tools = read(tool_path)
        service = read(service_path)
        assert core["prediction_count"] == 1720
        assert core["all_raw_API_and_usage_contracts_independently_verified"] is True
        assert tools["actual_model_provenance_verified"] is True
        t = tools["quality"]
        assert t["tasks"] == 32
        if "scores" in t:
            assert len(t["scores"]) == 32
            assert t["correct"] == sum(row["task_correct"] for row in t["scores"])
        else:
            # Earlier public summaries intentionally omit per-task records;
            # their full private audits remain bound by the published digest.
            assert tools["all_original_tasks_retained"] is True
            assert len(tools["source_actual_audit_sha256"]) == 64
        assert 0 <= t["correct"] <= 32
        assert t["task_success_percent"] == 100*t["correct"]/32
        assert t["all_original_tasks_retained_in_denominator"] is True
        identity = [core[key] for key in ("canonical_API_payloads_sha256", "labels_sha256", "scoring_protocol_sha256", "service")]
        if common is None:
            common = identity
        assert identity == common
        quality.append(dict(format=label, run_id=core["run_id"], python_source=core["python_source_commit"],
                            GSM8K=core["GSM8K_flexible_EM_percent"],
                            LongBench_macro=core["LongBench_core_macro_percent_unrounded"],
                            RepoBench_P=core["tasks"]["core::repobench-p"]["score_percent_unrounded"],
                            tool_correct=t["correct"], tool_denominator=32))
        per_task.extend(dict(format=label, task=name, **row) for name, row in core["tasks"].items())
        assert set(service["cases"]) == set(CASES)
        for name in CASES:
            case = service["cases"][name]
            profile, application = case["profile"], case["application"]
            assert profile["profiling_records"] == (64 if name == "govreport-c32" else 512)
            row = dict(format=label, case=name, run_id=service["run_id"],
                       audit_sha256=SOURCES[str(service_path.relative_to(HERE.parent))],
                       raw_manifest_sha256=service["SHA_manifest_sha256"])
            row.update({key: value for key, value in profile.items() if not isinstance(value, (dict, list))})
            row.update({"application_" + key: value for key, value in application.items()
                        if not isinstance(value, (dict, list))})
            for key, value in application["ranges"].items():
                if value is not None:
                    row.update({"sampled_" + key + "_" + statistic: number for statistic, number in value.items()})
            row["empty_or_whitespace_answers"] = case["Goodput"]["empty_or_whitespace_answers"]
            row["external_warmup_records"] = case["external_warmup"]["profiling_records"]
            services.append(row)
            assert len(case["Goodput"]["points"]) == 16
            curves.extend(dict(format=label, case=name, **point) for point in case["Goodput"]["points"])
    candidate = quality[-1]
    gates = {}
    for reference in quality[:2]:
        margins = dict(GSM8K=.5, LongBench_macro=1., RepoBench_P=2., tool_correct=1)
        gates[reference["format"]] = {
            key: {"delta": candidate[key] - reference[key], "minimum_delta": -margin,
                  "point_pass": candidate[key] - reference[key] >= -margin}
            for key, margin in margins.items()}
    for filename, rows in [("quality_exact.csv", quality), ("quality_tasks_exact.csv", per_task),
                           ("service_exact.csv", services), ("goodput_all16_exact.csv", curves)]:
        csv_new(HERE / filename, rows)
    result = dict(schema="repaired-mxfp4-complete-exact-export-v1", source_sha256=SOURCES,
                  formats=len(quality), service_cases=len(services), goodput_rows=len(curves),
                  frozen_engineering_point_gates=gates, statistical_noninferiority_automatically_claimed=False,
                  historical_MXFP4_has_Graph_correctness_confounder=True)
    with (HERE / "exact_provenance.json").open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"quality_rows": len(quality), "service_rows": len(services), "Goodput_rows": len(curves)}))


if __name__ == "__main__":
    main()
