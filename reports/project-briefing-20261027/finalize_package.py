#!/usr/bin/env python3
"""仅在本新报告目录内嵌表、生成导航/来源SHA并做静态核查。"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
OPS = "reports/operations/20261026T075846Z/"
EXP = "reports/experiments/"

KEY_SOURCES = [
    ("生产目标", "ops/PRODUCTION_DECISION_STANDARD.md", "真实请求、现代后端、质量与SLO、条件化支持；旧要求覆盖规则"),
    ("现代ShareGPT两配对", OPS + "modern_BF16_ShareGPT_ABBA_comparison_v1.json", "rows[*].metrics/delta_pct；duration_weighted_observation_pool；frozen_assets_identical"),
    ("现代ShareGPT第一对raw审计", OPS + "bf16_xqa_reference_b_independent_audit1.json", "metrics；allocation；封存与source/payload/SSE验证"),
    ("现代ShareGPT反序raw审计", OPS + "reverse_sharegpt_independent_audit_v2.json", "metrics；Graph/原payload/usage/驻留与终态"),
    ("现代ToolAgent两点", OPS + "modern_BF16_ToolAgent_pressure_comparison_v1.json", "points[*].arms.measurement/completion_timing/application_observations；跨点冻结核对"),
    ("ToolAgent .125 raw审计", OPS + "toolagent_s0125_independent_audit_v1.json", "arms；1689正式/399warm、source clock/payload/OSL/SSE与allocation"),
    ("ToolAgent .25 raw审计", OPS + "toolagent_s025_independent_audit_v1.json", "arms；1689正式/399warm、drain、同输出、Graph/bvar"),
    ("C128完整质量raw审计", OPS + "production_C128_quality_independent_audit_b1.json", "task_scores；paired_flips；model_weights_sha256；production_configuration；quality_acceptance"),
    ("C128质量及C8敏感性", OPS + "production_C128_quality_analysis_b1.json", "task_deltas；longbench_macro_delta；C128_minus_C8_*；bootstrap及finish"),
    ("新BF16 C8参考", OPS + "BF16_XQA_full_quality_independent_audit_a1.json", "1719原始预测/权重/模型Graph/官方重评分"),
    ("新BF16对保留FP8 C8", OPS + "BF16_XQA_full_quality_existing_FP8_comparison_a1.json", "同题跨run分数/配置绑定；FP8保持80+1639缓存边界"),
    ("原生工具258 raw审计", OPS + "native_tool_buffered_independent_audit_b1.json", "arms.*.metrics/official_argument_matches；paired_task_outcomes；checker及异常参考"),
    ("原生工具有限cohort分析", OPS + "native_tool_buffered_descriptive_analysis_b1.json", "arms；changes_pct；length_outlier；latency_changes_pct"),
    ("streaming失败原证据", OPS + "native_tool_stream_failure_a_independent_evidence_v1.json", "BF16首warm失败，无AIPerf/FP8；原FAILED保留"),
    ("streaming键排序CPU缺陷", OPS + "native_tool_stream_CPU_late_key_defect_v2.json", "原C++解析/flush复现；合法分块与旧坏wire，原token顺序未导出"),
    ("组件诊断raw审计", OPS + "component_diagnostic_independent_audit_c1.json", "116文件SHA、637请求/arm、3窗口/48forward、CUDA/Graph时间与phase"),
    ("组件描述性汇总", OPS + "component_diagnostic_descriptive_summary_c1.json", "arms.*.component_kernel_sum_ms/writer_share_of_all_kernel_time；matched_recorded_batch_metadata"),
    ("组件活动归属", OPS + "component_diagnostic_activity_times_c2.json", "CPU scope、kernel sum、busy union、Graph与未归属边界"),
    ("prefill工作区算子", OPS + "prefill_workspace_independent_audit_a1.json", "timings.raw_ms/median_ms；workspace_reserved_bytes；model_loaded=false/AIPerf_ran=false"),
    ("真实K/V c恢复审计", OPS + "range_probe_c_independent_recovered_audit_v1.json", "distribution；original_root_status；worker_measurement_seal_present；rate_clock_precision_verification"),
    ("真实K/V摘要", EXP + "production-kv-range-probe-20261027/audit_summary.json", "1536group、finite值/absmax/clipped/underflow/relative_l2、reset/原FAILED"),
    ("writer覆盖与下溢界", OPS + "range_probe_c_coverage_and_error_mechanism_v1.json", "214986=143037−2048+74125−128；underflow_energy_bounds"),
    ("scale预导出规则", OPS + "range_probe_candidate_scale_rule_before_export_v1.json", "先冻结判定规则，后查看真实分布"),
    ("scale保持决定", OPS + "range_probe_c_derived_scalar_proposal_v1.json", "保持K=.125/V=.0625，无新global scale"),
    ("BF16 Q新备选算子", OPS + "bf16q_decode_operator_independent_audit_a1.json", "graph128.dynamic_live_rows_and_context/backend；installed_sources_sha256；无模型/质量/服务"),
    ("BF16 Q原终态/占卡", OPS + "bf16q_decode_status_a4.json", "status=COMPLETE；files.result.json.value.allocation_intervals_ns；gpu_release_verified"),
    ("历史四臂归档", "reports/operations/20261026T040426Z/provenance_readonly.json", "historical_quality.files.summary.json.task_scores/longbench_8_task_macro；原始summary/CSV SHA"),
    ("ceff18case/九条件审计", EXP + "ceff-takeover-20261026T040426Z-v2/audit.json", "runs；case_count；graph/存储/hash与limitations"),
    ("ceff九配对原CSV", EXP + "ceff-takeover-20261026T040426Z-v2/pairs.csv", "同FA2自然停止、实际token、吞吐/ITL/TTFT变化"),
    ("早期FA3全量质量", EXP + "production-fa3-quality-full1719-20261026/audit.json", "80+1639合并、完整官方重评分、质量差/抽样区间"),
    ("旧XQA全量质量", EXP + "production-xqa-quality-full1719-20261026/audit.json", "三run恢复合并1719；同FP8保留预测、旧参考"),
    ("旧原速ToolAgent raw审计", OPS + "toolagent_trace_independent_audit_c2_py312.json", "arms；全部2088尝试、正式1689成功/拒绝、source/clock/usage"),
    ("旧BurstGPT raw审计", OPS + "burstgpt_f_independent_audit1_py312.json", "arms；6385全部/6221正式，offered限制"),
    ("旧conversation raw审计", OPS + "conversation_f_independent_audit1_py312.json", "原CONFLICT保留，独立恢复measurement与完整时钟"),
    ("旧synthetic raw审计", OPS + "synthetic_f_independent_audit1_py312.json", "过载成功/拒绝与较慢生成期，完整原始记录"),
    ("内容/末SSE计时", OPS + "STREAM_TIMING_FINDINGS.md", "实际AIPerf源码、14556条成功测量核对、非TCP EOF"),
    ("trace源语义", OPS + "PRODUCTION_TRACE_FINDINGS.md", "原长度/到达/prefix来源，匿名内容，过滤和BurstGPT eligible窗口"),
    ("源码机制", OPS + "PRODUCTION_SOURCE_FINDINGS.md", "BF16/FP8连续化、writer、固定scalar、非完整校准、代码评分"),
    ("上游14直接文件", OPS + "flashinfer_sm90_direct_headers_comparison_v1.json", "现装/固定上游14文件字节SHA一致，不证明全后端性能"),
    ("新候选质量计划", "plans/experiments/production-fp8-bf16q-full-quality-v1.md", "仅新FP8候选完整1719/C128，保留BF16参考；第45原模型前FAILED，无新质量结果"),
    ("新候选质量a失败", OPS + "bf16q_quality_status_a2.json", "files.result.json原FAILED/SHA与allocation_intervals_ns；driver释放，无模型质量；本地b未派发"),
    ("FA3 pilot输入manifest", "results/mirrors/prod-quality-fa3-pilot80-r0-20261026b-c1/files/support/agent_work/runs/prod-quality-fa3-pilot80-r0-20261026b/input/pilot/pilot_manifest.json", "samples_per_arm=80；GSM64；八任务各2、最短和最长各1；输入准备不是质量结果"),
    ("XQA pilot输入manifest", "results/mirrors/prod-quality-xqa-pilot80-r0-20261026c-c1/files/support/agent_work/runs/prod-quality-xqa-pilot80-r0-20261026c/input/pilot/pilot_manifest.json", "selection_rule/selected_source_indices与FA3 pilot完全一致，64+8×2=80"),
    ("第45独立失败审计", OPS + "bf16q_quality_a_independent_failure_evidence_v1.json", "72文件/20427910B逐SHA、零预测/无measurement seal；actual_GPU_hours与原interval一致，原FAILED保留"),
    ("本地b完整CPU dry-run", OPS + "bf16q_quality_b_full_actual_dry_run_CPU_v1.json", "source acc0ed6b；原full1719/C128实际dry-run、frozen policy有效；GPU_started=false，不计新模型结果"),
    ("本地b源码检查", OPS + "bf16q_quality_source_CPU_v2.log", "85 passed；仅本地CPU检查"),
    ("本地b控制器检查", OPS + "bf16q_quality_controller_CPU_v2.log", "Ran207/OK；仅本地CPU检查"),
    ("第46候选完整原始审计", OPS + "bf16q_full1719_quality_independent_audit_b1.json", "原COMPLETE/driver释放；1719原题/权重/Graph/官方重评分；actual_gpu_hours；audited_source_files_sha256"),
    ("第46三配置逐题质量对照", OPS + "bf16q_full1719_quality_comparison_b1.json", "task_scores/task_deltas；LongBench_macro；paired_bootstrap_95ci；paired_score_changes/cap_finish_counts；quality_accepted=false"),
    ("第46有限buffered cohort时钟", OPS + "bf16q_full1719_quality_cohort_latency_b1.json", "tasks.*.metrics/paired_descriptive_changes；new_GPU_runs=0；不同run/GPU/缓存，非AIPerf稳态"),
    ("第46正式报告摘要", EXP + "production-bf16q-quality-20261027/audit_summary.json", "canonical_audit/comparison/cohort_clocks三项SHA；123文件/44462952B；extra_service_matrix_stopped=true"),
    ("第46现成质量图manifest", EXP + "production-bf16q-quality-20261027/figures-v1/manifest.json", "comparison/generator/SVG/PNG/CSV SHA；固定原题相对现代BF16差值，非Q单因子"),
    ("第46完整gzip镜像", "results/mirrors/prod-quality-bf16q-full1719-c128-r1-20261027b-terminal-b2-gzip/files/SNAPSHOT.json", "123文件/44462952B逐SHA；原终态及所有封存质量证据"),
    ("第46xz取证失败", "results/mirrors/prod-quality-bf16q-full1719-c128-r1-20261027b-terminal-b1/transport.json", "rc255/keepalive无响应、partial不当有效raw、remote_writes=0；不是GPU失败"),
    ("第46gzip取证恢复", "results/mirrors/prod-quality-bf16q-full1719-c128-r1-20261027b-terminal-b2-gzip/transport.json", "rc0/gzip、客户端keepalive2→6/10秒；不重跑GPU"),
    ("第46官方评分逻辑兼容", OPS + "bf16q_quality_scoring_compatibility_CPU_v1.json", "only_new_strict_profile_literal_diff；evaluator SHA与candidate/reference audit绑定"),
]

NOTES = {
    **{i: "旧同FA2配对的单次run；六run共九条件/18case，不是现代生产参考。" for i in range(1, 7)},
    7: "旧binary失败；不能与ceff FP8拼配对。", 8: "旧binary不完整，旧单臂保留。",
    9: "四臂1719/arm历史汇总档案SHA一致；未独立重评分旧prediction。",
    10: "因最高目标转生产代表性而停止；无有效完整配对结论。",
    11: "首次FA3-Q候选ShareGPT测量恢复；原CONFLICT保留，RPS约+18.71%。",
    12: "FA3 pilot80/arm，仅接口与明显风险；不能宣称1719质量。",
    13: "FA3剩余1639/arm；与pilot精确合并1719，非新增1719题。",
    14: "XQA NaN尾页算子失败；没有模型/pilot。",
    15: "padded参考夹具页表storage别名错误；没有模型/pilot。",
    16: "XQA c算子/Graph/pilot80完成；全量仍需补1639。",
    17: "FP8剩余1639完整恢复，原CONFLICT；BF16缺臂，不重跑成功预测。",
    18: "仅补缺失BF161639，三run合成1719完整质量。",
    19: "prepare未启动、独立absence核验，0GPU-h，非重复。",
    20: "prepare未启动、独立absence核验，0GPU-h，非重复。",
    21: "旧FA3-decode BF16参考下ShareGPT；+152.58%不能冒称现代增益。",
    22: "ToolAgent prepare未启动，0GPU-h。",
    23: "冻结policy不匹配拒绝；没有正式trace测量。",
    24: "旧参考原速匿名ToolAgent2088；正式271→528/1689成功。",
    25: "旧参考BurstGPT6385；正式6221/6221均成功，无吞吐增量。",
    26: "旧参考原conversation，末响应后8.901s冲突；只恢复测量不改COMPLETE。",
    27: "旧参考原synthetic；多完成但生成期中位更慢。",
    28: "新BF16 XQA算子通过后归属保护停止；无服务measurement。",
    29: "现代ShareGPT第一配对，BF16→FP8，同GPU，正式512/arm。",
    30: "现代ToolAgent s=.125，2088全成功；正式1689、无吞吐增益，尾延迟更慢。",
    31: "现代ToolAgent s=.25，2088全成功；TTFT改善、生成期更慢。",
    32: "新BF16 C8单臂完整1719，与已保留FP8跨run比较；未新跑FP8。",
    33: "现代ShareGPT反序FP8→BF16，同GPU同528来源；第二paired run。",
    34: "C128双臂完整1719官方重评分；LB−1.7705pp，未获质量接受。",
    35: "组件prepare压缩白名单拒绝于mkdir前，0GPU-h。",
    36: "profiler annotation类别误计；14B前失败，费用计入。",
    37: "组件c每臂637请求/3窗口/48forward；机制诊断，不是headline吞吐重复。",
    38: "stream工具BF16首warm参数JSON失败；无AIPerf/FP8。",
    39: "buffered258+16warm；182正确/arm、7up7down，FP8单条重复到cap完整保留。",
    40: "prefill BF16工作区算子数值/Graph通过；约.6%收益与额外1.5GiB，停止全模型扩大。",
    41: "range a显式warmup0被CLI拒绝；无正式128请求/模型统计。",
    42: "range b原128结果独立恢复；Unicode拆行错误，统计导出前清理，不能给scale。",
    43: "range c原FAILED；ISO精度误拒绝，128结果/1536统计独立恢复，不重跑GPU。",
    44: "新BF16 Q decode仅算子/动态Graph128独立完整性验收通过；无14B/AIPerf/质量或服务收益。",
    45: "原FAILED_BEFORE_MODEL_PLAN_ENVIRONMENT、driver释放；72文件独立失败审计通过，零预测/无measurement。实际.001074001823888889GPU-h已结算，累计9.494151518055556；截至45 reserved0，本地b不计结果。",
}


def digest(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def rel_link(path: str) -> str:
    return "../../" + path


def local_targets() -> set[Path]:
    targets: set[Path] = set()
    for p in OUT.glob("*.md"):
        text = p.read_text(encoding="utf-8")
        for match in re.finditer(r"\]\(([^)]+)\)", text):
            raw = match.group(1).strip("<>")
            if raw.startswith(("http://", "https://", "mailto:", "#")):
                continue
            raw = unquote(raw.split("#", 1)[0])
            if not raw:
                continue
            targets.add((p.parent / raw).resolve())
    return targets


def main() -> None:
    # 主agent新原终态晚于在线INDEX/STATE时，不改源状态，用带SHA覆盖记录说明。
    failure_path = ROOT / (OPS + "bf16q_quality_status_a2.json")
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    result = failure["files"]["result.json"]
    assert failure["status"] == result["value"]["status"] == "FAILED"
    assert result["value"]["gpu_release_verified"] is True
    assert result["sha256"] == "aa8cddcc024aeaa0d0fe60e8d43341506f2e88ded66bd5b8ca825e16ba314086"
    duration_ns = sum(x["end_ns"] - x["start_ns"] for x in result["value"]["allocation_intervals_ns"])
    independent_path = ROOT / (OPS + "bf16q_quality_a_independent_failure_evidence_v1.json")
    independent = json.loads(independent_path.read_text(encoding="utf-8"))
    assert independent["terminal_result_sha256"] == result["sha256"]
    assert independent["model_loaded"] is False and independent["model_predictions"] == 0
    assert independent["worker_measurement_seal"] is False
    assert abs(independent["actual_GPU_hours"] - duration_ns / 3.6e12) < 1e-12
    cutoff = {
        "schema": "kv-project-briefing-cutoff-v2",
        "latest_run": "prod-quality-bf16q-full1719-c128-r0-20261027a",
        "latest_terminal": "FAILED_BEFORE_MODEL_PLAN_ENVIRONMENT",
        "original_root_status": "FAILED",
        "original_terminal_sha256": result["sha256"],
        "status_evidence": OPS + "bf16q_quality_status_a2.json",
        "status_evidence_sha256": digest(failure_path),
        "independent_failure_audit": OPS + "bf16q_quality_a_independent_failure_evidence_v1.json",
        "independent_failure_audit_sha256": digest(independent_path),
        "failure_files_SHA_verified": independent["files_SHA_verified"],
        "failure_bytes_SHA_verified": independent["bytes_SHA_verified"],
        "gpu_release_verified": True,
        "observed_allocation_ns": duration_ns,
        "observed_GPU_hours_from_raw_intervals": duration_ns / 3.6e12,
        "formal_ledger_settlement_for_run45": "settled",
        "settled_GPU_hours_run45": independent["actual_GPU_hours"],
        "settled_GPU_hours_through_run44_rounded": 9.4930775161,
        "settled_GPU_hours_through_run45": 9.494151518055556,
        "ledger_total_provenance": "Main agent explicit final settlement; raw run45 interval independently matches failure audit",
        "reservation_GPU_hours_as_of_run45_settlement": 0,
        "active_or_UNKNOWN_runs_as_of_run45_settlement": [],
        "prior_reservation_GPU_hours_not_consumed": 1.5,
        "quality_result_present": False,
        "cause_evidence_level": "confirmed_independent_failure_audit",
        "cause": independent["reason"],
        "new_b_scope": "local_preparation_at_run45_cutoff_no_GPU_or_model_result_no_run46_added",
        "local_b_source": "acc0ed6bf8bcd4500d8e5260cec17c2b5c67a9e9",
        "local_b_source_CPU_passed": 85,
        "local_b_controller_CPU_passed": 207,
        "local_b_actual_full1719_dry_run": OPS + "bf16q_quality_b_full_actual_dry_run_CPU_v1.json",
        "mutable_snapshot_policy": "Preserve original 45-row INDEX/STATE snapshot; later settlement/independent failure audit supplements run45 only, not subsequent runs."
    }
    original_cutoff_path = OUT / "data/BRIEFING_CUTOFF.json"
    if original_cutoff_path.is_file():
        assert json.loads(original_cutoff_path.read_text(encoding="utf-8")) == cutoff
    else:
        original_cutoff_path.write_text(json.dumps(cutoff, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    post45_path = OUT / "data/POST45_EVIDENCE_RUN46.json"
    post45 = json.loads(post45_path.read_text(encoding="utf-8")) if post45_path.is_file() else None
    post45_checks = json.loads((OUT / "data/POST45_checks.json").read_text(encoding="utf-8")) if post45 else None
    if post45:
        assert post45["run_id"] == "prod-quality-bf16q-full1719-c128-r1-20261027b" and post45_checks["status"] == "PASS"
        frozen = json.loads((OUT / "data/POST45_FROZEN_ORIGINALS.json").read_text(encoding="utf-8"))
        for record in frozen["files"]:
            p = OUT / record["path"]
            assert p.stat().st_size == record["bytes"] and digest(p) == record["sha256"], record["path"]
    # 用现有生成表替换一次性占位；不对项目原文件做任何写入。
    inserted = 0
    for p in OUT.glob("*.md"):
        text = p.read_text(encoding="utf-8")
        def substitute(m):
            nonlocal inserted
            inserted += 1
            return (OUT / "tables" / f"{m.group(1)}.md").read_text(encoding="utf-8").rstrip()
        new = re.sub(r"<!-- TABLE:([a-z0-9_]+) -->", substitute, text)
        if new != text:
            p.write_text(new, encoding="utf-8")

    index = list(csv.DictReader(io.StringIO((OUT / "data/INDEX_SNAPSHOT.csv").read_text(encoding="utf-8"))))
    assert len(index) == 45
    assert "bf16q-decode-op" in index[43]["run_id"]
    assert "prod-quality-bf16q-full1719" in index[44]["run_id"]
    inv = ["# 全部45条run的证据分类", "", "这是本报告冻结的索引快照，每行是一个运行/恢复/未启动/待验记录；不是45个成功实验，也不是45份独立用户样本。现代主服务、质量和机制结论见各专题，原状态与failed/conflict完整保留。后续新增任务不在此快照中。", "", "来源：[45行原始CSV快照](data/INDEX_SNAPSHOT.csv)，保留原完整source/binary/dataset/GPU/manifest/validity/notes字段。下面以中文说明能计入什么结论。", "", "| 行 | run_id | 原索引状态 | 汇报解释 |", "| --- | --- | --- | --- |"]
    for i, row in enumerate(index, 1):
        metric = row["metric_path"]
        status = row["status"]
        if metric and not metric.startswith("/") and (ROOT / metric).is_file():
            status = f"[{status}]({rel_link(metric)})"
        if i == 45:
            status = f"[FAILED_BEFORE_MODEL_PLAN_ENVIRONMENT]({rel_link(OPS + 'bf16q_quality_status_a2.json')})；索引快照原文：{row['status']}"
        inv.append(f"| {i} | {row['run_id']} | {status} | {NOTES[i]} |")
    inv.extend(["", "第44条已完成实际算子与独立完整性审计，以[最新独立审计](../operations/20261026T075846Z/bf16q_decode_operator_independent_audit_a1.json)为准，没有模型质量/服务收益。第45条原FAILED/driver释放、72文件独立失败审计与实际费用均已核对结算，以[独立失败审计](../operations/20261026T075846Z/bf16q_quality_a_independent_failure_evidence_v1.json)和[截止覆盖记录](data/BRIEFING_CUTOFF.json)为准。原45行快照保持，当时本地b的CPU准备不计模型结果；若有第46后续，单独列于下节。", "", "历史未启动/失败没有被删除，不把probe修复a/b/c当统计重复；未重新运行任何成功GPU实验。本汇报仅生成本新目录的文档、附表与来源核查。", ""])
    if post45:
        inv.extend(["## 原45条之后：第46单独补充", "", "原45行CSV/STATE与原16表保持字节不变，第46不混入原快照。新b修复a的模型前失败，未重跑成功预测，不增加独立用户。", "", "| 后续序号 | run_id | 最新独立证据 | 决策及限制 |", "| --- | --- | --- | --- |", "| 46 | prod-quality-bf16q-full1719-c128-r1-20261027b | [原COMPLETE/1719完整重评分独立验收](../../reports/operations/20261026T075846Z/bf16q_full1719_quality_independent_audit_b1.json) | 未观察整体质量修复；停止该BF16Q_FA2候选额外服务GPU矩阵，无新AIPerf、非Q单因子 |", "", "[第46单行原索引摘录](data/POST45_INDEX_RUN46.csv)可能保留取证时pending文字，以更晚完整audit为准；[06完整专题](06_第46条BF16Q完整质量与停止决定.md)、[后续证据/成本](data/POST45_EVIDENCE_RUN46.json)、[封存UTC与下次续接边界](data/REPORT_SEAL.json)。仅在用户再要求时从run46之后续接，不自动跟踪新实验。", ""])
    (OUT / "RUN_INVENTORY.md").write_text("\n".join(inv), encoding="utf-8")

    source_lines = ["# 来源、身份与核查", "", "数字优先取既有独立raw审计及其已验收派生汇总；本汇报未重新运行GPU、未重评分模型预测，也未将worker完成当作独立验收。16张附表由[本地提取脚本](build_evidence_tables.py)从原JSON/CSV生成，同名CSV放在data/，正文嵌入相同表。", "", "[完整来源manifest](data/source_manifest.json)逐文件给出仓库相对路径、字节与SHA-256；[报告产物manifest](data/artifact_manifest.json)给出本次文档/表格/脚本的SHA。SHA只认证文件身份，不表示科学结论自动通过。动态INDEX/STATE按读取时冻结，实时状态可能继续变化。", "", "## 关键身份", "", "| 身份 | 值/边界 |", "| --- | --- |", "| 服务源码 | 1c2be15976b7fbb1d8b79e469659b2c8e52cff00 |", "| 正式C128质量源码 | 11e7cddbfe33eb21d209c7f48f5e461ac56545eb；推理引擎与服务版逐字节一致 |", "| 复用C++ binary SHA-256 | 277bd8f391fbef978c86d77a976d488d567b0781bf19c0cc8a9c6763fccb21f3 |", "| 质量实际逐shard模型identity | cf67b8e03cf159dd6017e36da84c28e143e59ce766261db2f6c97aabca7aae73 |", "| ShareGPT冻结输入SHA-256 | 810415584dc2c751f0cf0e895cd7c4f1418ad338c03f70781d9abfc4e0df44a1 |", "| ToolAgent源内容SHA-256 | 48a2db1a13d3bc05e6330140c64f604ba366df20d3c9e128b5c35a01c1fa5f71 |", "| 原生BFCL buffered输入SHA-256 | 47b4b1229449514dba089ab6f6de60a156e1f1df81b5e516b83fc493a2efbcbf |", "| heldout128范围输入SHA-256 | ca61bbf5c47e491179c5749cfe5a6c67044daf84d643885e8ae87178d4141869 |", "| 新BF16 Q decode源提交 | 78004505e2db44e88994899e8ea3e25c109978a7；仅算子证据 |", "| 当前主要runtime | Torch2.10.0+cu130、Triton3.6.0、Transformers5.3.0、FlashInfer0.6.18.post1、AIPerf0.13.0；以各manifest实际记录为准 |", "", "以上身份由下表对应审计/索引绑定；性能run未逐次重新hash全部权重，不能声称每次均有全权重内容审计。各run不可变request/terminal/arm seal、mirror每文件SHA由原审计保存，本汇报的manifest不重复复制所有巨大raw。", "", "## 关键来源与字段", "", "| ID | 支持的结论 | 精确来源 | 关键字段/口径 | SHA-256 |", "| --- | --- | --- | --- | --- |"]
    if post45:
        source_lines[2] = "数字优先取既有独立raw审计及已验收汇总。原16张表/CSV与45行INDEX/STATE保持原字节，由[原提取器](build_evidence_tables.py)派生；第46另加4表、27行完整buffered时钟CSV，由[补充生成器](build_post45_supplement.py)读取原审计复核。共20张表，本汇报未运行GPU、生成或重评分模型预测。"
        source_lines[4] = "[完整来源manifest](data/source_manifest.json)分列原45与第46输入SHA；[产物manifest](data/artifact_manifest.json)覆盖本目录文档/表格/CSV/脚本。SHA认证身份，不自动表示质量接受。原45冻结层与[第46补充](06_第46条BF16Q完整质量与停止决定.md)分开，当前报告仅封存至run46，UTC及下次续接规则见[封存记录](data/REPORT_SEAL.json)。"
    for i, (label, path, fields) in enumerate(KEY_SOURCES, 1):
        p = ROOT / path
        assert p.is_file(), path
        source_lines.append(f"| S{i:02} | {label} | [{p.name}]({rel_link(path)}) | {fields} | {digest(p)} |")
    base_numeric_check_count = len(json.loads((OUT / "data/numeric_checks.json").read_text(encoding="utf-8"))["checks"])
    post45_numeric_check_count = len(post45_checks["checks"]) if post45 else 0
    numeric_check_count = base_numeric_check_count + post45_numeric_check_count
    numeric_summary = f"原[数值核查](data/numeric_checks.json)保留{base_numeric_check_count}组，覆盖既有服务/容量/质量/工具/组件/工作区/范围/Graph与pilot、失败及CPU准备。"
    cutoff_note = "如源STATE/INDEX后续变化，以原45行快照及明确的截止说明为准；本报告不是在线状态页，项目未标记全完成。"
    if post45:
        numeric_summary += f"第46[补充核查](data/POST45_checks.json)新增{post45_numeric_check_count}组：三配置原题/分数/差值/逐题/cap、既有bootstrap口径、27行cohort分母与实际token、123原镜像SHA、终态/ceil微秒成本、取证恢复和35个原文件字节不变；另验证正式报告/既有图SHA，合计{numeric_check_count}组。"
        cutoff_note = "原45行INDEX/STATE/CUTOFF与原16表/CSV的35文件SHA保持不变；第46完整质量未整体修复、停止该候选额外服务GPU矩阵。报告封存至run46，后续仅在用户要求时从run46之后续接；不主动跟踪或更新，UTC见封存记录。"
    source_lines.extend(["", "## 核查方式与结果", "", numeric_summary, "", "[静态核查结果](data/validation.json)给出正文/来源链接、canonical SHA绑定、占位符及输入身份检查。独立完整性/官方评分沿用已验收审计；本汇报只复核已评分值与文件，没有独立重复GPU数值或模型预测。", "", cutoff_note, ""])
    (OUT / "SOURCES.md").write_text("\n".join(source_lines), encoding="utf-8")

    # 收集正文可点击的原证据及主要审计；明确区分文件与目录。
    targets = local_targets()
    missing = sorted(str(p) for p in targets if not p.exists())
    # manifest/validation等会在本函数稍后生成，不把预期输出误报缺失。
    generated = {OUT / "data/source_manifest.json", OUT / "data/artifact_manifest.json", OUT / "data/validation.json", OUT / "data/REPORT_SEAL.json"}
    missing = [s for s in missing if Path(s) not in generated]
    assert not missing, missing
    targets |= {(ROOT / p).resolve() for _, p, _ in KEY_SOURCES}
    extracted = json.loads((OUT / "data/extraction_sources.json").read_text(encoding="utf-8"))
    extracted_map = {r["repo_relative_path"]: r for r in extracted}
    targets |= {(ROOT / p).resolve() for p in extracted_map}
    post45_extracted_map = {}
    if post45:
        post45_sources = json.loads((OUT / "data/POST45_sources.json").read_text(encoding="utf-8"))["sources"]
        post45_extracted_map = {r["repo_relative_path"]: r for r in post45_sources}
        targets |= {(ROOT / p).resolve() for p in post45_extracted_map}
    source_records, changed, changed_post45 = [], [], []
    for p in sorted(targets):
        if p.is_relative_to(OUT):
            continue
        repo_relative = str(p.relative_to(ROOT))
        if not p.is_file():
            source_records.append({"repo_relative_path": repo_relative, "is_directory": True})
            continue
        record = {"repo_relative_path": repo_relative, "bytes": p.stat().st_size, "sha256_at_finalize": digest(p)}
        if repo_relative in extracted_map:
            orig = extracted_map[repo_relative]
            record["sha256_at_extract"] = orig["sha256"]
            record["bytes_at_extract"] = orig["bytes"]
            record["changed_since_extract"] = record["sha256_at_finalize"] != orig["sha256"]
            if record["changed_since_extract"]:
                changed.append(repo_relative)
        if repo_relative in post45_extracted_map:
            orig = post45_extracted_map[repo_relative]
            record["sha256_at_post45_extract"] = orig["sha256"]
            record["bytes_at_post45_extract"] = orig["bytes"]
            record["changed_since_post45_extract"] = record["sha256_at_finalize"] != orig["sha256"]
            if record["changed_since_post45_extract"]:
                changed_post45.append(repo_relative)
        source_records.append(record)
    (OUT / "data/source_manifest.json").write_text(json.dumps({"schema": "kv-project-briefing-source-manifest-v2", "created_utc": datetime.now(timezone.utc).isoformat(), "scope": "original45_snapshot_plus_run46_supplement_local_readonly_no_GPU_no_new_predictions", "report_cutoff_run": 46 if post45 else 45, "sources": source_records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 摘要里声明的canonical SHA必须和实际输入一致。
    sha_checks = []
    for f in sorted(p for p in targets if p.name == "audit_summary.json" and p.is_file()):
        a = json.loads(f.read_text(encoding="utf-8"))
        pairs = []
        for pathkey, hashkey in (("canonical_audit", "canonical_audit_sha256"), ("canonical_audit", "audit_sha256"), ("full_independent_audit_path", "full_independent_audit_sha256"), ("canonical_comparison", "sha256")):
            if pathkey in a and hashkey in a:
                pairs.append((a[pathkey], a[hashkey]))
        for path, expected in pairs:
            target = ROOT / path
            assert target.is_file(), path
            actual = digest(target)
            assert actual == expected, (f, path, actual, expected)
            sha_checks.append({"summary": str(f.relative_to(ROOT)), "canonical": path, "sha256": actual})
    if post45:
        for binding in post45_checks["canonical_SHA_bindings"]:
            assert digest(ROOT / binding["canonical"]) == binding["sha256"]
            sha_checks.append({"supplement_check": binding["label"], "canonical": binding["canonical"], "sha256": binding["sha256"]})

    for p in OUT.glob("*.md"):
        assert "<!-- TABLE:" not in p.read_text(encoding="utf-8"), p
    validation = {"status": "PASS", "created_utc": datetime.now(timezone.utc).isoformat(), "report_cutoff_run": 46 if post45 else 45, "index_snapshot_rows": len(index), "post45_run_rows": 1 if post45 else 0, "original_frozen_files_verified": len(frozen["files"]) if post45 else None, "table_count": len(list((OUT / "tables").glob("*.md"))), "base_numeric_check_groups": base_numeric_check_count, "post45_numeric_check_groups": post45_numeric_check_count, "numeric_check_groups": numeric_check_count, "local_link_targets": len(local_targets()), "missing_local_links": [], "source_files_or_directories": len(source_records), "canonical_audit_SHA_checks": sha_checks, "extracted_sources_changed_at_finalize": changed, "post45_extracted_sources_changed_at_finalize": changed_post45, "note_on_mutable_sources": "原45行INDEX/STATE快照不扩展；第46单独补充，更晚源状态不代表扩展截止；静态独立审计保持SHA。", "new_GPU_runs": 0, "new_model_predictions": 0, "package_writer_outputs_scoped_to_this_directory": True}
    # 除在线状态/索引外，提取输入不得被静默替换。
    assert not [p for p in changed if p not in ("state/STATE.json", "results/INDEX.csv")], changed
    assert not [p for p in changed_post45 if p not in ("state/STATE.json", "results/INDEX.csv")], changed_post45
    if post45:
        for record in frozen["files"]:
            f = OUT / record["path"]
            assert f.stat().st_size == record["bytes"] and digest(f) == record["sha256"]
        seal = {"schema": "kv-project-briefing-seal-v1", "report_cutoff_run": 46, "report_cutoff_run_id": post45["run_id"], "sealed_utc": datetime.now(timezone.utc).isoformat(), "evidence_status": "independent_complete_quality_comparison_and_stopped_extra_service_GPU_matrix", "quality_accepted": False, "original_45_index_and_state_bytes_unchanged": True, "original_frozen_files_verified": len(frozen["files"]), "tables": validation["table_count"], "numeric_check_groups": numeric_check_count, "next_report_starts_after_run": 46, "automatic_report_updates": False, "future_updates_require_user_request": True, "source_manifest": "data/source_manifest.json", "source_manifest_sha256": digest(OUT / "data/source_manifest.json"), "validation": "data/validation.json", "artifact_manifest": "data/artifact_manifest.json", "main_experiments_continue_independently": True}
        (OUT / "data/REPORT_SEAL.json").write_text(json.dumps(seal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "data/validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    artifacts = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p != OUT / "data/artifact_manifest.json":
            artifacts.append({"path": str(p.relative_to(OUT)), "bytes": p.stat().st_size, "sha256": digest(p)})
    (OUT / "data/artifact_manifest.json").write_text(json.dumps({"schema": "kv-project-briefing-artifact-manifest-v1", "artifacts": artifacts, "self_excluded": True}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "report_cutoff_run": 46 if post45 else 45, "sealed_utc": seal["sealed_utc"] if post45 else None, "tables_embedded_this_call": inserted, "documents": len(list(OUT.glob("*.md"))), "source_records": len(source_records), "local_link_targets": len(local_targets()), "canonical_SHA_checks": len(sha_checks), "artifact_files": len(artifacts), "numeric_check_groups": numeric_check_count, "changed_extraction_sources": changed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
