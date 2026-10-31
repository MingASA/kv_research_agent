#!/usr/bin/env python3
"""从已验收的本地摘要生成汇报附表；不运行模型、不重评分、不改既有结果。"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
TABLES = OUT / "tables"
DATA = OUT / "data"
OPS = "reports/operations/20261026T075846Z/"
EXP = "reports/experiments/"
SOURCES: dict[str, dict] = {}
CHECKS: list[str] = []


def read(path: str) -> bytes:
    p = ROOT / path
    b = p.read_bytes()
    SOURCES[path] = {"repo_relative_path": path, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
    return b


def js(path: str):
    return json.loads(read(path))


def rows(path: str):
    return list(csv.DictReader(io.StringIO(read(path).decode("utf-8"))))


def table(name: str, headers: list[str], body: list[list]) -> None:
    assert all(len(r) == len(headers) for r in body), name
    text = "| " + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers) + " |\n"
    text += "".join("| " + " | ".join(str(x) for x in row) + " |\n" for row in body)
    (TABLES / f"{name}.md").write_text(text, encoding="utf-8")
    with (DATA / f"{name}.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(body)


def near(a: float, b: float, tolerance: float = 1e-8) -> None:
    assert abs(a - b) <= tolerance * max(1, abs(a), abs(b)), (a, b)


def fmt(v: float, places: int = 3) -> str:
    return f"{v:.{places}f}"


def main() -> None:
    TABLES.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    sg = js(OPS + "modern_BF16_ShareGPT_ABBA_comparison_v1.json")
    sg_summary = js(EXP + "production-bf16-xqa-sharegpt-replicates-20261027/audit_summary.json")
    assert sg == sg_summary
    assert sg["independent_paired_runs"] == 2 and sg["frozen_assets_identical"] is True
    body = []
    sg_detail = []
    for number, r in enumerate(sg["rows"], 1):
        for arm in ("bf16", "fp8_e4m3"):
            m = r["metrics"][arm]
            near(m["request_rps"], 512 / m["observation_seconds"])
            near(m["output_tps"], m["actual_output_tokens"] / m["observation_seconds"])
            near(m["osl_mean"], m["actual_output_tokens"] / 512)
            assert m["profiling_records"] == 512 and m["warmup_records"] == 16
            assert sum(m["finish_reason_counts"].values()) == 512
            body.append([number, "→".join(r["arm_order"]), arm, "512/512", fmt(m["observation_seconds"], 6), fmt(m["request_rps"], 6), m["actual_output_tokens"], fmt(m["output_tps"], 3), fmt(m["osl_mean"], 3), f'{m["finish_reason_counts"]["stop"]}/{m["finish_reason_counts"]["length"]}'])
        for key in ("request_rps", "output_tps", "osl_mean", "ttft_p95_ms", "itl_p50_ms", "e2e_p95_ms"):
            near(r["delta_pct"][key], 100 * (r["metrics"]["fp8_e4m3"][key] / r["metrics"]["bf16"][key] - 1))
    table("sharegpt_runs", ["配对", "实际臂序", "arm", "测量成功/尝试", "observation s", "req/s", "实际输出 tokens", "实际输出 tok/s", "输出均值 tokens", "stop/length"], body)
    r = sg["rows"][0]
    for label, key in [("TTFT P50 ms", "ttft_p50_ms"), ("TTFT P95 ms", "ttft_p95_ms"), ("TTFT P99 ms", "ttft_p99_ms"), ("逐请求平均 ITL P50 ms", "itl_p50_ms"), ("逐请求平均 ITL P95 ms", "itl_p95_ms"), ("逐请求平均 ITL P99 ms", "itl_p99_ms"), ("E2E P50 ms", "e2e_p50_ms"), ("E2E P95 ms", "e2e_p95_ms"), ("E2E P99 ms", "e2e_p99_ms")]:
        sg_detail.append([label, fmt(r["metrics"]["bf16"][key]), fmt(r["metrics"]["fp8_e4m3"][key]), f'{r["delta_pct"][key]:+.3f}%'])
    table("sharegpt_first_pair_latency", ["指标", "现代 BF16", "FP8", "FP8 相对变化"], sg_detail)
    pool = sg["duration_weighted_observation_pool"]
    for arm in pool:
        near(pool[arm]["request_rps"], pool[arm]["requests_sum"] / pool[arm]["observed_seconds_sum"])
        near(pool[arm]["output_tps"], pool[arm]["actual_output_tokens_sum"] / pool[arm]["observed_seconds_sum"])
    near(sg["duration_weighted_request_rps_delta_pct"], 100 * (pool["fp8_e4m3"]["request_rps"] / pool["bf16"]["request_rps"] - 1))
    near(sg["duration_weighted_output_tps_delta_pct"], 100 * (pool["fp8_e4m3"]["output_tps"] / pool["bf16"]["output_tps"] - 1))
    CHECKS.append("ShareGPT: 4臂各512测量+16warm、输出sum/均值/finish、observation吞吐、两run时长汇总与所有列示变化一致")

    pressure = js(OPS + "modern_BF16_ToolAgent_pressure_comparison_v1.json")
    pressure_summary = js(EXP + "production-bf16-xqa-toolagent-pressure-20261027/audit_summary.json")
    assert pressure["points"] == pressure_summary["points"]
    p_service, p_latency, p_scheduler, p_qualified = [], [], [], []
    for p in pressure["points"]:
        for arm, a in p["arms"].items():
            m = a["measurement"]
            assert (m["attempts"], m["successes"], m["errors"], m["actual_output_tokens"]) == (1689, 1689, 0, 304758)
            assert a["warm_cache"]["attempts"] == a["warm_cache"]["successes"] == 399
            near(m["completion_rps_including_drain"], 1689 / m["elapsed_including_drain_seconds"])
            p_service.append([p["arrival_scaling"], arm, fmt(m["offered_rps"], 7), "1689/1689", fmt(a["drain_seconds"], 6), fmt(m["elapsed_including_drain_seconds"], 6), fmt(m["completion_rps_including_drain"], 6), m["actual_output_tokens"], fmt(m["actual_output_tokens"] / m["elapsed_including_drain_seconds"], 6)])
            c = a["completion_timing"]
            p_latency.append([p["arrival_scaling"], arm, fmt(m["offered_ttft_ms"]["p50"] / 1000), fmt(m["offered_ttft_ms"]["p95"] / 1000), fmt(m["offered_ttft_ms"]["p99"] / 1000), fmt(m["itl_ms"]["p50"]), fmt(c["final_SSE_inclusive_per_server_output_token_ms"]["p50"]), fmt(m["offered_e2e_ms"]["p50"] / 1000), fmt(m["offered_e2e_ms"]["p95"] / 1000), fmt(m["offered_e2e_ms"]["p99"] / 1000)])
            app = a["application_observations"]
            g = app["gauges"]
            p_scheduler.append([p["arrival_scaling"], arm, fmt(g["num_running_requests"]["p50"], 0), fmt(g["num_running_requests"]["max"], 0), fmt(g["num_waiting_requests"]["p50"], 0), fmt(g["num_waiting_requests"]["max"], 0), fmt(g["kv_cache_utilization_perc"]["p50"], 6), fmt(app["sample_bounded_prefix_hit_token_fraction"] * 100, 3)])
            q = c["illustrative_TTFT_and_final_SSE_inclusive_mean100ms_qualified_RPS"]
            p_qualified.append([p["arrival_scaling"], arm, fmt(q["1"], 6), fmt(q["5"], 6), fmt(q["10"], 6)])
    table("toolagent_pressure_service", ["时钟 s", "arm", "offered req/s", "测量成功/尝试", "drain s", "测量+drain s", "完成 req/s", "实际输出 tokens", "实际输出 tok/s"], p_service)
    table("toolagent_pressure_latency", ["时钟 s", "arm", "offered TTFT P50 s", "P95 s", "P99 s", "官方 ITL P50 ms", "末 SSE 完成/token P50 ms", "完整 E2E P50 s", "P95 s", "P99 s"], p_latency)
    table("toolagent_pressure_scheduler", ["时钟 s", "arm", "running P50", "running max", "waiting P50", "waiting max", "used 比例 P50", "采样 prefix token hit %"], p_scheduler)
    table("toolagent_illustrative_qualified", ["时钟 s", "arm", "TTFT≤1s 合格 req/s", "≤5s 合格 req/s", "≤10s 合格 req/s"], p_qualified)
    CHECKS.append("ToolAgent现代两点: 四臂1689测量/399warm完整、304758实际输出、含drain吞吐重算一致；完成/token与官方ITL分列")

    a0 = pressure["points"][0]["arms"]
    storage_rows = []
    for arm, a in a0.items():
        v = a["allocation"]
        assert v["token_capacity"] == v["pages_per_layer"] * 128
        assert v["resident_bytes"] == v["token_capacity"] * v["bytes_per_token_per_layer"] * 48
        storage_rows.append([arm, v["bytes_per_token_per_layer"], v["pages_per_layer"], v["token_capacity"], v["token_capacity"] - 128, v["resident_bytes"], fmt(v["resident_bytes"] / 2**30, 6)])
    table("resident_capacity", ["arm", "B/token/layer（含scale）", "physical pages/layer", "物理 token容量（含page0）", "扣page0 token容量", "48层 resident B", "48层 resident GiB"], storage_rows)
    near(100 * (1 - 2112 / 4096), 48.4375)
    CHECKS.append("resident: 128token/page、48层总字节、page0扣除、48.4375%压缩均按整数核对")

    trace_service, trace_latency = [], []
    for family in ("toolagent", "burstgpt", "conversation", "synthetic"):
        d = js(EXP + f"production-xqa-{family}-original-20261027/audit_summary.json")
        for arm, a in d["measurement_arms"].items():
            metrics = a["metrics"]
            m = metrics["phases"]["measurement"]
            assert m["successes"] + m["errors"] == m["attempts"]
            near(m["completion_rps_including_drain"], m["successes"] / m["elapsed_including_drain_seconds"])
            trace_service.append([family, arm, m["attempts"], m["successes"], fmt(100 * m["successes"] / m["attempts"]), m["errors"], fmt(m["offered_rps"], 6), fmt(m["completion_rps_including_drain"], 6), m["actual_output_tokens"], fmt(m["actual_output_tokens"] / m["elapsed_including_drain_seconds"], 3), fmt(metrics["drain_seconds_after_offered_window"], 3)])
            trace_latency.append([family, arm, fmt(m["offered_ttft_ms"]["p50"] / 1000), fmt(m["offered_ttft_ms"]["p95"] / 1000), fmt(m["itl_ms"]["p50"]), fmt(m["offered_e2e_ms"]["p95"] / 1000)])
    table("original_traces_service", ["匿名家族", "arm", "测量尝试", "成功", "成功率 %", "拒绝", "offered req/s", "完成 req/s（含drain）", "实际输出 tokens", "实际输出 tok/s", "drain s"], trace_service)
    table("original_traces_latency", ["匿名家族", "arm", "offered TTFT P50 s", "P95 s", "官方 ITL P50 ms", "完整 E2E P95 s"], trace_latency)
    CHECKS.append("原四trace: 全部测量分母=成功+拒绝，完成/实际token吞吐以各自含drain时间复算一致")

    q128 = js(EXP + "production-bf16-xqa-quality-c128-20261027/audit_summary.json")
    qs = rows(EXP + "production-bf16-xqa-quality-c128-20261027/task_scores.csv")
    body = []
    for r in qs:
        b, f, delta = float(r["bf16"]), float(r["fp8_e4m3"]), float(r["delta_percentage_points"])
        near(delta, 100 * (f - b))
        body.append([r["metric"], "1319" if "GSM" in r["metric"] else ("8×50" if "macro" in r["metric"].lower() else "50"), fmt(b * 100, 4), fmt(f * 100, 4), f"{delta:+.4f}"])
    table("quality_c128", ["指标", "独立题目/arm", "现代 BF16 分数 %", "FP8 分数 %", "差值 pp"], body)
    CHECKS.append("C128质量: 全表百分比/百分点换算一致；1719为1319+8×50，每题配对，不把bootstrap视为GPU重复")

    native = js(OPS + "native_tool_buffered_descriptive_analysis_b1.json")
    native_audit = js(OPS + "native_tool_buffered_independent_audit_b1.json")
    assert native["audit_sha256"] == SOURCES[OPS + "native_tool_buffered_independent_audit_b1.json"]["sha256"]
    nr = []
    for a, x in native["arms"].items():
        near(x["request_rps"], 258 / x["observation_seconds"])
        near(x["argument_match_rps"], 182 / x["observation_seconds"])
        near(x["actual_output_tps"], x["output_tokens"] / x["observation_seconds"])
        nr.append([a, "258/258", "182/258", fmt(x["accuracy"] * 100, 4), fmt(x["observation_seconds"], 6), fmt(x["request_rps"], 4), fmt(x["argument_match_rps"], 4), x["output_tokens"], fmt(x["actual_output_tps"], 3), x["length_cap_count"]])
    table("native_tools_service", ["arm", "服务成功/尝试", "官方匹配", "匹配率 %", "observation s", "服务 req/s", "匹配正确 req/s", "实际输出 tokens", "实际输出 tok/s", "length次数"], nr)
    assert sum(native["paired_task_outcomes"].values()) == 258
    CHECKS.append("原生工具: 258分母与175/69/7/7配对一致，182匹配正确req/s与整批服务/token吞吐分母一致")

    component = js(OPS + "component_diagnostic_descriptive_summary_c1.json")
    cr = []
    for a, x in component["arms"].items():
        near(x["writer_share_of_all_kernel_time"], x["component_kernel_sum_ms"]["kv_write"] / x["cuda_kernel_sum_ms"])
        cr.append([a, x["prefill_forwards"], x["prefill_input_tokens"], fmt(x["component_kernel_sum_ms"]["prefill_attention"]), fmt(x["component_kernel_sum_ms"]["kv_write"]), fmt(100 * x["writer_share_of_all_kernel_time"]), x["Graph_decode_forwards"], fmt(x["decode_kernel_ms_per_forward_median"])])
    table("component", ["arm", "prefill批次", "query tokens", "prefill attention累计 ms", "writer累计 ms", "writer占kernel累计 %", "Graph decode批次", "Graph整模型/批中位 ms"], cr)
    CHECKS.append("组件诊断: writer份额使用各臂全部kernel累计时间，17/18prefill与query总量分开保留")

    workspace = js(EXP + "production-prefill-workspace-20261027/audit_summary.json")
    wr = []
    for k, x in workspace["timings"].items():
        b, f = x["median_ms"]["resident_FP8_FA2"], x["median_ms"]["resident_FP8_FA3_BF16_workspace"]
        near(x["candidate_median_reduction_pct"], 100 * (1 - f / b))
        wr.append([k, fmt(b, 6), fmt(f, 6), fmt(x["candidate_median_reduction_pct"], 4)])
    table("prefill_workspace", ["口径", "直接FP8 FA2 ms", "BF16工作区FA3 ms", "中位降幅 %"], wr)
    assert workspace["workspace_reserved_bytes"] == 1536 * 2**20 + 3 * 2**20 + 4
    assert workspace["model_loaded"] is False and workspace["AIPerf_ran"] is False
    CHECKS.append("prefill工作区: 两种中位降幅与1613758468B拆分一致；无模型/AIPerf声明保留")

    probe = js(EXP + "production-kv-range-probe-20261027/audit_summary.json")
    coverage = js(OPS + "range_probe_c_coverage_and_error_mechanism_v1.json")
    body = []
    for k, v in probe["distribution"]["aggregate"].items():
        body.append([k, fmt(v["absmax"], 3), "56" if "key" in k else "28", v["clipped"], fmt(v["underflow_pct"], 6), fmt(v["relative_l2"] * 100, 4)])
    table("kv_range", ["元数据phase/KV", "absmax", "固定scale覆盖上限", "饱和数", "非零→零 %", "QDQ相对L2 %"], body)
    assert coverage["actual_prompt_tokens"] - coverage["actual_cached_prompt_tokens"] + coverage["actual_output_tokens"] - coverage["requests"] == coverage["observed_valid_writer_tokens_per_layer"] == 214986
    assert probe["original_root_status"] == "FAILED" and probe["worker_measurement_seal_present"] is False
    CHECKS.append("K/V范围: 214986=143037−2048+74125−128，1536分组摘要；原FAILED/无worker seal保持")

    legacy = js("reports/operations/20261026T040426Z/provenance_readonly.json")["historical_quality"]["files"]["summary.json"]
    legacy_scores = {}
    for x in legacy["task_scores"]:
        legacy_scores.setdefault(x["task"], {})[x["arm"]] = x["scores"]
    lr = []
    for task in legacy_scores:
        for metric in (("strict_em", "flexible_em") if task == "gsm8k" else ("score",)):
            values = []
            for a in ("auto", "int8", "fp8_e4m3", "int4"):
                key = f"{a}_{metric}" if task == "gsm8k" else a
                values.append(fmt(100 * legacy_scores[task][a][key], 4))
            lr.append([f"{task} {metric}" if task == "gsm8k" else task] + values)
    lr.append(["LongBench 8任务macro"] + [fmt(legacy["longbench_8_task_macro"][a] * 100, 4) for a in ("auto", "int8", "fp8_e4m3", "int4")])
    table("legacy_four_arm_quality", ["历史指标（仅档案汇总，%）", "auto", "INT8", "FP8", "INT4"], lr)
    ceff = rows(EXP + "ceff-takeover-20261026T040426Z-v2/pairs.csv")
    body = []
    for r in sorted(ceff, key=lambda x: (int(x["isl_nominal"]), int(x["osl_cap"]), int(x["concurrency"]))):
        for k in ("output_tps", "request_rps", "osl_mean", "itl_p50_ms", "ttft_p95_ms"):
            near(float(r[f"{k}_change_pct"]), 100 * (float(r[f"fp8_{k}"]) / float(r[f"bf16_{k}"]) - 1))
        body.append([r["isl_nominal"], r["osl_cap"], r["concurrency"], fmt(float(r["bf16_output_tps"]), 2), fmt(float(r["fp8_output_tps"]), 2), f'{float(r["output_tps_change_pct"]):+.2f}', f'{float(r["request_rps_change_pct"]):+.2f}', f'{float(r["itl_p50_ms_change_pct"]):+.2f}', f'{float(r["ttft_p95_ms_change_pct"]):+.2f}', f'{float(r["bf16_osl_mean"]):.1f}→{float(r["fp8_osl_mean"]):.1f}'])
    table("ceff_nine_conditions", ["名义ISL", "输出cap", "C", "BF16 tok/s", "FP8 tok/s", "tok/s变化 %", "req/s变化 %", "ITL P50变化 %", "TTFT P95变化 %", "实际输出均值"], body)
    CHECKS.append("历史ceff九条件: 所有表内变化从各自BF16/FP8原始数值重算一致；不拼接现代基线")

    decode = js(OPS + "bf16q_decode_operator_independent_audit_a1.json")
    decode_status = js(OPS + "bf16q_decode_status_a4.json")
    assert decode["model_loaded"] is False and decode["model_quality_certified"] is False
    assert decode["new_service_benefit_certified"] is False
    assert decode["graph128"]["writer_in_same_graph"] is True
    assert decode_status["status"] == "COMPLETE"
    assert len(decode["installed_sources_sha256"]) == 3
    for r in decode["graph128"]["dynamic_live_rows_and_context"]:
        assert r["bucket"] == 128 and r["relative_l2"] < .01
        assert r["live_rows"] + r["padding_outputs_discarded"] == 128
    intervals = decode_status["files"]["result.json"]["value"]["allocation_intervals_ns"]
    actual_ns = sum(r["end_ns"] - r["start_ns"] for r in intervals)
    near(actual_ns / 3.6e12, .042364935, tolerance=1e-9)
    CHECKS.append("BF16 Q备选: 55文件/369269B的算子完整性、4个动态Graph128行数与<.01门槛；无模型/质量/服务收益；GPU-h按原interval核对")

    pilots = []
    for run in ("prod-quality-fa3-pilot80-r0-20261026b", "prod-quality-xqa-pilot80-r0-20261026c"):
        p = f"results/mirrors/{run}-c1/files/support/agent_work/runs/{run}/input/pilot/pilot_manifest.json"
        pilot = js(p)
        assert pilot["samples_per_arm"] == 80
        assert len(pilot["selected_source_indices"]["gsm8k"]) == 64
        assert "shortest+longest prompt per task" in pilot["selection_rule"]
        assert all(len(v) == 2 for k, v in pilot["selected_source_indices"].items() if k != "gsm8k")
        assert sum(map(len, pilot["selected_source_indices"].values())) == 64 + 8 * 2 == 80
        pilots.append(pilot)
    assert pilots[0] == pilots[1]
    CHECKS.append("两个原pilot manifest一致: 每臂GSM64+八LongBench任务各2=80；每任务最短和最长各1条，不是各2条")

    failure = js(OPS + "bf16q_quality_a_independent_failure_evidence_v1.json")
    failure_status = js(OPS + "bf16q_quality_status_a2.json")
    original_result = failure_status["files"]["result.json"]
    assert failure["original_status"] == "FAILED" and failure["model_loaded"] is False
    assert failure["model_predictions"] == 0 and failure["worker_measurement_seal"] is False
    assert (failure["files_SHA_verified"], failure["bytes_SHA_verified"]) == (72, 20427910)
    assert failure["terminal_result_sha256"] == original_result["sha256"]
    actual = sum(x["end_ns"] - x["start_ns"] for x in original_result["value"]["allocation_intervals_ns"]) / 3.6e12
    near(failure["actual_GPU_hours"], actual, tolerance=1e-12)
    near(9.4930775161 + actual, 9.494151518055556, tolerance=1e-9)
    CHECKS.append("第45独立失败审计: 72文件/20427910B、零预测/无measurement seal，实际.001074001823888889GPU-h与原interval一致；已结算累计9.494151518055556，旧第44舍入尾差保留")

    dry = js(OPS + "bf16q_quality_b_full_actual_dry_run_CPU_v1.json")
    source_log = read(OPS + "bf16q_quality_source_CPU_v2.log").decode("utf-8")
    controller_log = read(OPS + "bf16q_quality_controller_CPU_v2.log").decode("utf-8")
    assert dry["GPU_started"] is False and dry["actual_frozen_policy_valid"] is True
    assert dry["source_commit"] == "acc0ed6bf8bcd4500d8e5260cec17c2b5c67a9e9"
    assert dry["plan"]["arms"] == ["fp8_e4m3"]
    assert sum(x["selected_rows"] for x in dry["plan"]["data_audit"]["datasets"].values()) == 1719
    assert "85 passed" in source_log and "Ran 207 tests" in controller_log and "OK" in controller_log
    CHECKS.append("本地b准备: source acc0ed6b、85源码/207控制器检查及实际完整1719 dry-run通过，GPU_started=false；不增加第46条或任何模型结果")

    # 修订汇报时保持原45条快照；主agent的后续INDEX/STATE不能扩入本汇报。
    snapshot = DATA / "INDEX_SNAPSHOT.csv"
    excerpt = DATA / "STATE_EXCERPT.json"
    if snapshot.is_file() and excerpt.is_file():
        index_bytes = snapshot.read_bytes()
        assert len(list(csv.DictReader(io.StringIO(index_bytes.decode("utf-8"))))) == 45
        prior_sources = json.loads((DATA / "extraction_sources.json").read_text(encoding="utf-8"))
        for source in prior_sources:
            if source["repo_relative_path"] in ("results/INDEX.csv", "state/STATE.json"):
                SOURCES[source["repo_relative_path"]] = source
    else:
        index_bytes = read("results/INDEX.csv")
        snapshot.write_bytes(index_bytes)
        state = js("state/STATE.json")
        safe_state = {k: state.get(k) for k in ("phase", "active_run_id", "active_run_ids", "last_checkpoint_utc", "last_evidence_utc")}
        safe_state["project_complete"] = state.get("current_work", {}).get("project_complete")
        safe_state["scope_note"] = "只保存安全最小摘录；旧current_work状态文本可能过时，主agent实时STATE优先。"
        excerpt.write_text(json.dumps(safe_state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (DATA / "extraction_sources.json").write_text(json.dumps(list(SOURCES.values()), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (DATA / "numeric_checks.json").write_text(json.dumps({"status": "PASS", "checks": CHECKS, "new_GPU_runs": 0, "new_model_predictions": 0, "rescored_existing_predictions": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"tables": len(list(TABLES.glob("*.md"))), "source_files": len(SOURCES), "numeric_checks": len(CHECKS), "index_rows": len(list(csv.DictReader(io.StringIO(index_bytes.decode('utf-8')))))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
