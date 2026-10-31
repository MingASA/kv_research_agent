#!/usr/bin/env python3
"""第46条的只读综合；原45条快照、原16表与其CSV保持字节不变。"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DATA = OUT / "data"
TABLES = OUT / "tables"
OPS = "reports/operations/20261026T075846Z/"
RUN = "prod-quality-bf16q-full1719-c128-r1-20261027b"
MIRROR = f"results/mirrors/{RUN}-terminal-b2-gzip/"
SOURCE_RECORDS: dict[str, dict] = {}
CHECKS: list[str] = []
SHA_BINDINGS: list[dict] = []
TASKS = ("gsm8k", "narrativeqa", "qasper", "multifieldqa_en", "2wikimqa", "hotpotqa", "musique", "lcc", "repobench-p")


def digest(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: str) -> bytes:
    p = ROOT / path
    b = p.read_bytes()
    SOURCE_RECORDS[path] = {"repo_relative_path": path, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
    return b


def js(path: str):
    return json.loads(read(path))


def near(a, b, tolerance=1e-11):
    assert abs(a - b) <= tolerance * max(1, abs(a), abs(b)), (a, b)


def fmt(x, digits=7):
    return f"{x:.{digits}f}"


def signed(x, digits=7):
    return f"{x:+.{digits}f}"


def table(name, headers, body):
    assert name.startswith("post46_")
    assert all(len(r) == len(headers) for r in body)
    s = "| " + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers) + " |\n"
    s += "".join("| " + " | ".join(map(str, r)) + " |\n" for r in body)
    (TABLES / f"{name}.md").write_text(s, encoding="utf-8")
    with (DATA / f"{name}.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(body)


def freeze_originals():
    p = DATA / "POST45_FROZEN_ORIGINALS.json"
    if p.is_file():
        frozen = json.loads(p.read_text(encoding="utf-8"))
    else:
        original_tables = [p for p in TABLES.glob("*.md") if not p.name.startswith("post46_")]
        assert len(original_tables) == 16
        paths = [DATA / "INDEX_SNAPSHOT.csv", DATA / "STATE_EXCERPT.json", DATA / "BRIEFING_CUTOFF.json"]
        paths += original_tables + [DATA / f"{p.stem}.csv" for p in original_tables]
        frozen = {"scope": "original45_snapshots_and_original16_tables_and_CSV", "files": [{"path": str(p.relative_to(OUT)), "bytes": p.stat().st_size, "sha256": digest(p)} for p in sorted(paths)]}
        p.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for record in frozen["files"]:
        f = OUT / record["path"]
        assert f.stat().st_size == record["bytes"] and digest(f) == record["sha256"], record["path"]
    assert len(list(csv.DictReader(io.StringIO((DATA / "INDEX_SNAPSHOT.csv").read_text(encoding="utf-8"))))) == 45
    return frozen


def bind(label, expected, path):
    actual = SOURCE_RECORDS[path]["sha256"]
    assert expected == actual, (label, expected, actual)
    SHA_BINDINGS.append({"label": label, "canonical": path, "sha256": actual})


def main():
    frozen = freeze_originals()
    candidate_path = OPS + "bf16q_full1719_quality_independent_audit_b1.json"
    reference_path = OPS + "production_C128_quality_independent_audit_b1.json"
    comparison_path = OPS + "bf16q_full1719_quality_comparison_b1.json"
    clocks_path = OPS + "bf16q_full1719_quality_cohort_latency_b1.json"
    scoring_path = OPS + "bf16q_quality_scoring_compatibility_CPU_v1.json"
    a, r, c, clocks, scoring = (js(p) for p in (candidate_path, reference_path, comparison_path, clocks_path, scoring_path))
    bind("comparison→candidate raw audit", c["candidate_audit_sha256"], candidate_path)
    bind("comparison→retained modern BF16/FP8-XQA audit", c["reference_audit_sha256"], reference_path)
    bind("comparison→official scoring AST proof", c["scoring_logic_proof_sha256"], scoring_path)
    bind("quality cohort clocks→quality comparison", clocks["comparison_sha256"], comparison_path)
    assert a["run_id"] == c["candidate_run_id"] == clocks["candidate_run_id"] == RUN
    assert a["root_complete"] and a["gpu_release_verified"] and a["terminal_status_preserved"] == "COMPLETE"
    assert a["audited_arms"] == ["fp8_e4m3"] and a["samples_per_arm"] == c["records"] == 1719
    assert c["quality_accepted"] is False and c["independent_new_users"] == 0
    assert clocks["new_GPU_runs"] == 0
    assert a["audited_source_commit"] == "acc0ed6bf8bcd4500d8e5260cec17c2b5c67a9e9"
    CHECKS.append("四个canonical SHA绑定一致；第46原COMPLETE/driver释放、仅新FP8候选1719、quality_accepted=false、无新独立用户；cohort时钟未新跑GPU")

    report_dir = "reports/experiments/production-bf16q-quality-20261027/"
    report_summary = js(report_dir + "audit_summary.json")
    bind("formal report summary→raw audit", report_summary["canonical_audit_sha256"], candidate_path)
    bind("formal report summary→comparison", report_summary["comparison_sha256"], comparison_path)
    bind("formal report summary→cohort clocks", report_summary["cohort_clocks_sha256"], clocks_path)
    assert report_summary["files_SHA_verified"] == 123 and report_summary["bytes_SHA_verified"] == 44462952
    assert report_summary["extra_service_matrix_stopped"] is True and report_summary["quality_accepted"] is False
    figure_dir = report_dir + "figures-v1/"
    figure_manifest = js(figure_dir + "manifest.json")
    bind("existing figure manifest→comparison", figure_manifest["comparison_sha256"], comparison_path)
    for name, expected in figure_manifest["files"].items():
        read(figure_dir + name)
        bind("existing quality figure/data output", expected, figure_dir + name)
    CHECKS.append("正式第46报告summary三项SHA与原JSON一致、停止额外服务矩阵且quality_accepted=false；复用现有SVG/PNG/数据CSV及其manifest，比较输入与3产物SHA全部一致，不复制或重画")

    for field in ("model_weights_sha256", "model_metadata_sha256", "versions", "protocol_files_sha256"):
        assert a[field] == r[field], field
    for field in ("binary_sha256", "model_config_sha256", "image_id", "service", "graph_mode", "fp8_scales", "concurrency"):
        assert a["production_configuration"][field] == r["production_configuration"][field], field
    assert a["production_configuration"]["concurrency"] == 128
    assert a["production_configuration"]["operators"]["fp8_e4m3"]["decode_internal_q"] == "bf16"
    source_diff = sorted(k for k in set(a["audited_source_files_sha256"]) | set(r["audited_source_files_sha256"]) if a["audited_source_files_sha256"].get(k) != r["audited_source_files_sha256"].get(k))
    assert source_diff == ["xllm/python/attention/flashinfer.py", "xllm/python/attention/native_profile.py"]
    assert scoring["only_new_strict_profile_literal_diff"] is True
    assert scoring["baseline_evaluator_sha256"] == r["evaluator_sha256"]
    assert scoring["candidate_evaluator_sha256"] == a["evaluator_sha256"]
    CHECKS.append("两run实际weights/metadata/versions/scorer协议及C128服务配置一致；引擎源码仅显式attention两个文件变化，评分逻辑AST证明绑定；不同kernel/run不能归因Q单因子")

    quality_rows, flip_rows = [], []
    macro_values = {arm: [] for arm in ("BF16", "FP8_XQA", "FP8_BF16Q")}
    for task in TASKS:
        n = 1319 if task == "gsm8k" else 50
        items = a["paired_items"][task]
        old_items = r["paired_items"][task]
        assert len(items) == len(old_items) == len(a["reuse_mapping"][task]) == n
        assert [(x["original_source_index"], x["id"]) for x in items] == [(x["original_source_index"], x["id"]) for x in old_items]
        metrics = ("strict_em", "flexible_em") if task == "gsm8k" else ("official_score",)
        for metric in metrics:
            values = c["task_scores"][task][metric]
            deltas = c["task_deltas"][task][metric]
            item_key = "flexible_scores" if metric == "flexible_em" else "scores"
            candidate_scores = [x[item_key]["fp8_e4m3"] for x in items]
            near(sum(candidate_scores) / n, values["FP8_BF16Q"])
            for arm, raw_arm in (("BF16", "bf16"), ("FP8_XQA", "fp8_e4m3")):
                old_scores = [x[item_key][raw_arm] for x in old_items]
                near(sum(old_scores) / n, values[arm])
                near((values["FP8_BF16Q"] - values[arm]) * 100, deltas[f"versus_{arm}_pp"])
                flips = Counter("up" if f > b else "down" if f < b else "tie" for f, b in zip(candidate_scores, old_scores))
                declared = c["paired_score_changes"][f"{task}_{metric}_versus_{arm}"]
                assert all(flips.get(k, 0) == declared.get(k, 0) for k in ("up", "down", "tie"))
            label = f"GSM8K {metric}" if task == "gsm8k" else task
            quality_rows.append([label, n, fmt(values["BF16"] * 100), fmt(values["FP8_XQA"] * 100), fmt(values["FP8_BF16Q"] * 100), signed(deltas["versus_BF16_pp"]), signed(deltas["versus_FP8_XQA_pp"])])
            cap = c["cap_finish_counts"]
            caps = [cap["reference_pair"]["bf16"][task], cap["reference_pair"]["fp8_e4m3"][task], cap["candidate"]["fp8_e4m3"][task]]
            desc = []
            for arm in ("BF16", "FP8_XQA"):
                v = c["paired_score_changes"][f"{task}_{metric}_versus_{arm}"]
                assert sum(v.values()) == n
                desc.append("/".join(str(v.get(k, 0)) for k in ("up", "down", "tie")))
            flip_rows.append([label, n, "/".join(map(str, caps)), *desc])
        if task != "gsm8k":
            for arm in macro_values:
                macro_values[arm].append(c["task_scores"][task]["official_score"][arm])
    for arm, vals in macro_values.items():
        assert len(vals) == 8
        near(sum(vals) / 8, c["LongBench_macro"][arm])
    macro = c["LongBench_macro"]
    quality_rows.append(["LongBench 8任务macro", "8×50", fmt(macro["BF16"] * 100), fmt(macro["FP8_XQA"] * 100), fmt(macro["FP8_BF16Q"] * 100), signed((macro["FP8_BF16Q"] - macro["BF16"]) * 100), signed((macro["FP8_BF16Q"] - macro["FP8_XQA"]) * 100)])
    table("post46_quality_scores", ["指标", "记录/arm", "现代BF16 %", "原FP8-XQA %", "新FP8-BF16Q %", "新−BF16 pp", "新−原FP8 pp"], quality_rows)
    table("post46_quality_flips_caps", ["指标", "记录/arm", "length次数 BF16/原FP8/新", "新vsBF16 上/下/持平", "新vs原FP8 上/下/持平"], flip_rows)
    CHECKS.append("原题源索引/ID逐条对齐；1719分数均值、全部任务百分点与逐题上/下/持平从两份已评分audit复核；8任务等权macro及cap计数一致，未重新评分或生成模型预测")

    assert c["bootstrap"]["draws"] == 5000 and c["bootstrap"]["seed"] == 20261026
    ci_rows = []
    for arm in ("BF16", "FP8_XQA"):
        values = c["paired_bootstrap_95ci"][f"versus_{arm}"]
        for metric, delta in (("LongBench_macro_delta_pp", (macro["FP8_BF16Q"] - macro[arm]) * 100), ("GSM_flexible_delta_pp", c["task_deltas"]["gsm8k"]["flexible_em"][f"versus_{arm}_pp"])):
            lo, hi = values[metric]
            assert lo < hi and math.isfinite(lo) and math.isfinite(hi)
            ci_rows.append(["LongBench macro" if metric.startswith("LongBench") else "GSM flexible", f"新−{arm}", signed(delta), f"[{signed(lo, 6)}, {signed(hi, 6)}]"])
    assert c["paired_bootstrap_95ci"]["versus_BF16"]["LongBench_macro_delta_pp"][1] < 0
    lo, hi = c["paired_bootstrap_95ci"]["versus_FP8_XQA"]["LongBench_macro_delta_pp"]
    assert lo < 0 < hi
    table("post46_quality_bootstrap", ["指标", "比较", "实测差 pp", "配对bootstrap 95%区间 pp"], ci_rows)
    CHECKS.append("既有5000次/seed20261026 bootstrap口径及单位核对；新vsBF16 LB区间在0下、新vs原FP8跨0；不重做bootstrap或据跨0证明等价")

    cohort_rows, full_clock_rows = [], []
    for task in TASKS:
        task_clocks = clocks["tasks"][task]
        metrics = task_clocks["metrics"]
        n = 1319 if task == "gsm8k" else 50
        for arm in ("BF16", "FP8_XQA", "FP8_BF16Q"):
            m = metrics[arm]
            assert m["completed_requests"] == n
            near(m["finite_cohort_completed_RPS"], n / m["first_request_to_last_response_seconds"])
            near(m["finite_cohort_actual_output_TPS"], m["actual_output_tokens"] / m["first_request_to_last_response_seconds"])
            near(m["mean_output_tokens"], m["actual_output_tokens"] / n)
            length_counts = c["cap_finish_counts"]["candidate"]["fp8_e4m3"] if arm == "FP8_BF16Q" else c["cap_finish_counts"]["reference_pair"]["bf16" if arm == "BF16" else "fp8_e4m3"]
            assert m["length_finish_count"] == length_counts[task]
            lat = m["client_complete_latency_seconds"]
            full_clock_rows.append([task, arm, n, m["first_request_to_last_response_seconds"], m["arrival_span_seconds"], m["actual_output_tokens"], m["finite_cohort_completed_RPS"], m["finite_cohort_actual_output_TPS"], m["mean_output_tokens"], m["mean_prompt_tokens"], lat["p50"], lat["p95"], lat["p99"], lat["max"], m["length_finish_count"]])
        deltas = []
        for arm in ("BF16", "FP8_XQA"):
            delta = 100 * (metrics["FP8_BF16Q"]["finite_cohort_completed_RPS"] / metrics[arm]["finite_cohort_completed_RPS"] - 1)
            near(delta, task_clocks["paired_descriptive_changes"][arm]["finite_cohort_RPS_change_pct"])
            deltas.append(signed(delta, 4))
        cohort_rows.append([task, n, fmt(metrics["BF16"]["finite_cohort_completed_RPS"], 6), fmt(metrics["FP8_XQA"]["finite_cohort_completed_RPS"], 6), fmt(metrics["FP8_BF16Q"]["finite_cohort_completed_RPS"], 6), *deltas])
    table("post46_cohort_rps", ["任务", "N", "BF16有限RPS", "原FP8有限RPS", "新BF16Q有限RPS", "新vsBF16 %", "新vs原FP8 %"], cohort_rows)
    with (DATA / "post46_cohort_all_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "arm", "N", "first_request_to_last_response_s", "arrival_span_s", "actual_output_tokens", "finite_RPS", "finite_actual_output_TPS", "mean_output_tokens", "mean_prompt_tokens", "complete_latency_p50_s", "complete_latency_p95_s", "complete_latency_p99_s", "complete_latency_max_s", "length_count"])
        w.writerows(full_clock_rows)
    CHECKS.append("9任务×3配置共27个有限buffered cohort：N/实际token/均值/length与原质量结果对应；RPS/TPS分母及全部相对变化重算一致，非AIPerf/稳态/TTFT/ITL")

    snapshot = js(MIRROR + "files/SNAPSHOT.json")
    assert len(snapshot["files"]) == 123
    assert sum(x["size"] for x in snapshot["files"]) == 44462952
    mirror_files = ROOT / (MIRROR + "files")
    for record in snapshot["files"]:
        f = mirror_files / record["relative_path"]
        assert f.resolve().is_relative_to(mirror_files.resolve()) and f.is_file() and not f.is_symlink()
        assert f.stat().st_size == record["size"] and digest(f) == record["sha256"]
    terminal_record = next(x for x in snapshot["files"] if x["source"].endswith(f"/{RUN}/result.json"))
    terminal_path = MIRROR + "files/" + terminal_record["relative_path"]
    terminal = js(terminal_path)
    bind("raw quality audit→original root terminal", a["terminal_result_sha256"], terminal_path)
    assert terminal["status"] == "COMPLETE" and terminal["gpu_release_verified"] and not terminal["live_owned_containers"]
    ns = sum(x["end_ns"] - x["start_ns"] for x in terminal["allocation_intervals_ns"])
    near(ns / 3.6e12, a["actual_gpu_hours"])
    billed_us = (ns + 999) // 1000
    billed_GPU_hours = billed_us / 3.6e9
    previous_total = json.loads((DATA / "BRIEFING_CUTOFF.json").read_text(encoding="utf-8"))["settled_GPU_hours_through_run45"]
    total = 9.724063727777779
    near(previous_total + billed_GPU_hours, total, tolerance=1e-12)
    CHECKS.append("成功gzip镜像123文件/44462952B全部本地SHA再次核对；root原COMPLETE/driver释放与audit终态SHA一致；实际GPU-h按ns核对，ceil微秒结算与累计9.724063727777779一致")

    failed_transport = js(f"results/mirrors/{RUN}-terminal-b1/transport.json")
    good_transport = js(MIRROR + "transport.json")
    assert failed_transport["rc"] == 255 and failed_transport["archive_codec"] == "xz"
    assert "not responding" in failed_transport["stderr"]
    assert good_transport["rc"] == 0 and good_transport["archive_codec"] == "gzip"
    assert failed_transport["remote_writes"] == good_transport["remote_writes"] == 0
    assert "ServerAliveCountMax=2" in failed_transport["command"] and "ServerAliveCountMax=6" in good_transport["command"]
    assert "ServerAliveInterval=10" in good_transport["command"]
    CHECKS.append("取证xz/SSH rc255与gzip rc0分开保留；均remote_writes=0、keepalive2→6/10s；失败partial不当有效raw，不增加GPU运行或模型失败")

    index_bytes = read("results/INDEX.csv")
    all_rows = list(csv.DictReader(io.StringIO(index_bytes.decode("utf-8"))))
    selected = [x for x in all_rows if x["run_id"] == RUN]
    assert len(selected) == 1
    row = selected[0]
    assert row["git_sha"] == a["audited_source_commit"] and row["concurrency"] == "128"
    with (DATA / "POST45_INDEX_RUN46.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)
    freeze_originals()
    CHECKS.append("原45行INDEX/STATE/CUTOFF及原16表/CSV共35文件SHA保持不变；第46只做单行独立摘录，原索引若仍pending由更晚完整audit解释")

    evidence = {
        "schema": "kv-project-briefing-post45-run46-v1",
        "scope": "supplement_to_original_45_row_snapshot_only_run46",
        "run_id": RUN,
        "original_terminal": "COMPLETE",
        "terminal_sha256": a["terminal_result_sha256"],
        "classification": c["classification"],
        "independent_integrity_and_official_rescoring_passed": True,
        "candidate_quality_accepted": False,
        "samples_per_candidate_arm": 1719,
        "independent_new_users": 0,
        "source_commit": a["audited_source_commit"],
        "runtime_sources_digest": a["runtime_sources_digest"],
        "explicit_attention_source_changes": source_diff,
        "candidate_audit": candidate_path,
        "comparison": comparison_path,
        "finite_buffered_cohort_clocks": clocks_path,
        "source_index_row_status_preserved": row["status"],
        "source_index_pending_text_does_not_override_newer_audit": True,
        "actual_gpu_hours_ns_interval": a["actual_gpu_hours"],
        "billing_interval_ceil_microseconds": billed_us,
        "billed_GPU_hours_from_ceil_microseconds": billed_GPU_hours,
        "settled_GPU_hours_through_run45": previous_total,
        "settled_GPU_hours_through_run46": total,
        "reserved_GPU_hours_at_run46_settlement": 0,
        "active_or_UNKNOWN_runs_at_run46_settlement": [],
        "new_candidate_AIPerf_service_run": False,
        "decision": "stop_extra_service_GPU_matrix_for_this_BF16Q_FA2_candidate_no_overall_quality_repair_observed",
        "decision_provenance": "Main agent explicit production decision after independent comparison; root persistence belongs to main agent",
        "BF16_default_unchanged": True,
        "original_FP8_XQA_conditional_service_results_retained": True,
        "Q_single_factor_causality_established": False,
        "next_scope": "CPU_only_verify_long_document_code_heldout_KV_range_evidence_and_deployable_recipes; proposal_not_a_new_GPU_or_scale_result"
    }
    (DATA / "POST45_EVIDENCE_RUN46.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (DATA / "POST45_sources.json").write_text(json.dumps({"scope": "post45_run46", "sources": list(SOURCE_RECORDS.values())}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (DATA / "POST45_checks.json").write_text(json.dumps({"status": "PASS", "checks": CHECKS, "canonical_SHA_bindings": SHA_BINDINGS, "original_frozen_files_verified": len(frozen["files"]), "new_tables": 4, "new_cohort_metric_rows": 27, "new_GPU_runs_by_briefing_agent": 0, "new_model_predictions_by_briefing_agent": 0, "rescored_model_predictions_by_briefing_agent": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "new_tables": 4, "new_numeric_checks": len(CHECKS), "source_files": len(SOURCE_RECORDS), "original_frozen_files_unchanged": len(frozen["files"]), "run46_actual_GPU_h": a["actual_gpu_hours"], "through46_settled_GPU_h": total}, ensure_ascii=False))


if __name__ == "__main__":
    main()
