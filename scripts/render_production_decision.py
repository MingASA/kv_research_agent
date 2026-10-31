#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Render exportable figures from verified production comparisons."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(serving_path: Path, pressure_path: Path, quality_path: Path, out: Path) -> dict:
    serving, pressure, quality = [json.loads(p.read_text()) for p in
                                  (serving_path, pressure_path, quality_path)]
    if (serving['classification'] != 'confirmed_two_sameGPU_opposite_order_deployment_pairs'
            or pressure['classification'] != 'confirmed_same_payload_runtime_two_pressure_observations'
            or quality['classification'] != 'confirmed_C128_quality_tradeoff_and_C8_descriptive_sensitivity'):
        raise ValueError('Require verified production comparisons')
    if out.exists():
        raise ValueError('Output directory must be new')
    out.mkdir(parents=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    bf_color, fp_color = '#3975a8', '#d86c3b'
    data = []
    ax = axes[0, 0]
    x = np.arange(2)
    for offset, key, label, color in [(-.18, 'request_rps', 'Completed requests/s', bf_color),
                                      (.18, 'output_tps', 'Actual output tokens/s', fp_color)]:
        values = [row['delta_pct'][key] for row in serving['rows']]
        bars = ax.bar(x + offset, values, width=.32, label=label, color=color)
        ax.bar_label(bars, labels=[f'{v:.1f}%' for v in values], padding=3)
        for i, value in enumerate(values):
            data.append(('ShareGPT', f'pair{i+1}', key + '_FP8_delta_pct', value))
    ax.set_xticks(x, ['BF16 then FP8', 'FP8 then BF16'])
    ax.set_ylabel('FP8 relative improvement (%)')
    ax.set_ylim(0, max(row['delta_pct']['request_rps'] for row in serving['rows']) * 1.28)
    ax.set_title('A. Natural-EOS ShareGPT, closed-loop C128', loc='left', fontweight='bold')
    ax.legend(loc='upper right', frameon=False, fontsize=8)
    ax.text(.02, -.23, 'Two observed same-GPU pairs; no population CI.\nSame 512 source requests reused, with different output lengths.',
            transform=ax.transAxes, fontsize=8, va='top')
    points = pressure['points']
    for ax, panel, metric, ylabel in [
            (axes[0, 1], 'B. ToolAgent offered TTFT P95', 'offered_ttft_p95_seconds', 'Seconds; lower is better'),
            (axes[1, 0], 'C. ToolAgent completion/token proxy P50', 'completion_proxy_p50_ms', 'ms per server output token; lower is better')]:
        for arm, label, color in [('bf16', 'BF16 FA3 + XQA', bf_color),
                                  ('fp8_e4m3', 'FP8 FA2 + XQA', fp_color)]:
            values = []
            for point in points:
                a = point['arms'][arm]
                value = (a['measurement']['offered_ttft_ms']['p95'] / 1000 if metric.startswith('offered') else
                         a['completion_timing']['final_SSE_inclusive_per_server_output_token_ms']['p50'])
                values.append(value)
                data.append(('ToolAgent', str(point['offered_rps']), arm + '_' + metric, value))
            ax.plot([p['offered_rps'] for p in points], values, marker='o', color=color, label=label)
            for point, value in zip(points, values):
                ax.annotate(f'{value:.2f}', (point['offered_rps'], value), xytext=(4, 5),
                            textcoords='offset points', fontsize=8, color=color)
        ax.set_title(panel, loc='left', fontweight='bold')
        ax.set_xlabel('Offered requests/s (same 1,689 measurement requests)')
        ax.set_xticks([p['offered_rps'] for p in points])
        ax.set_xticklabels([f'{p["offered_rps"]:.3f}' for p in points])
        ax.set_ylabel(ylabel)
        ax.set_xlim(points[0]['offered_rps'] - .12, points[-1]['offered_rps'] + .18)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=8)
        if metric.startswith('offered'):
            note = 'Both arms: 100% success at both loads; completion RPS +0.0% / +0.8%.\nOne pair/load, different GPUs across loads; no exact capacity threshold.'
        else:
            note = 'First content to last SSE divided by (server output count - 1).\nNot hardware decode-step timing; anonymous length/prefix/arrival replay.'
        ax.text(.02, -.26, note, transform=ax.transAxes, fontsize=8, va='top')
    ax = axes[1, 1]
    tasks = ['narrativeqa', 'qasper', 'multifieldqa_en', '2wikimqa', 'hotpotqa', 'musique', 'lcc', 'repobench-p']
    values = [quality['task_deltas'][task] * 100 for task in tasks]
    bars = ax.barh(tasks, values, color=['#b04f4f' if v < 0 else '#478965' for v in values])
    ax.bar_label(bars, labels=[f'{v:+.2f}' for v in values], padding=3, fontsize=8)
    ax.axvline(0, color='#777777', linewidth=.8)
    ax.invert_yaxis()
    ax.set_xlim(min(min(values), 0) - 1.3, max(max(values), 0) + 1.3)
    ax.set_xlabel('FP8 minus BF16 (percentage points)')
    ax.set_title('D. Same full-cohort C128 quality pair', loc='left', fontweight='bold')
    gsm_delta = quality['task_deltas']['gsm8k']['flexible_em'] * 100
    macro_delta = quality['longbench_macro_delta'] * 100
    ax.text(.02, -.26, f'GSM8K flexible {gsm_delta:+.2f} pp; LongBench macro {macro_delta:+.2f} pp.\n50/task + 1,319 GSM8K; code similarity is not execution success.',
            transform=ax.transAxes, fontsize=8, va='top')
    for task, value in zip(tasks, values):
        data.append(('Quality_C128', task, 'FP8_minus_BF16_pp', value))
    for ax in axes.flat:
        ax.grid(axis='x' if ax is axes[1, 1] else 'y', color='#dddddd', alpha=.55)
        ax.set_axisbelow(True)
    fig.suptitle('xLLM KV quantization: workload-dependent production evidence', fontsize=16, y=.985)
    fig.text(.015, .015,
             'H800 / Qwen2.5-14B / resident FP8 KV, internal FP8-Q decode / production prefix + CUDA Graph.\n'
             'Different deployable kernels; no pure-dtype causal claim. Quality and SLO margins are not accepted.',
             fontsize=9)
    fig.subplots_adjust(left=.08, right=.97, top=.91, bottom=.15, hspace=.7, wspace=.37)
    for suffix in ('svg', 'png'):
        fig.savefig(out / ('production_evidence.' + suffix), dpi=180)
    plt.close(fig)
    with (out / 'plotted_values.csv').open('x', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(('workload', 'condition', 'metric', 'value'))
        writer.writerows(data)
    manifest = {'classification': 'derived_visualization_from_verified_measurements',
                'inputs': {str(p): _sha(p) for p in (serving_path, pressure_path, quality_path)},
                'generator_sha256': _sha(Path(__file__)), 'matplotlib_version': matplotlib.__version__,
                'artifacts': {p.name: _sha(p) for p in out.iterdir() if p.is_file()},
                'GPU_work': False, 'quality_accepted': False}
    with (out / 'manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('serving', 'pressure', 'quality', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = render(args.serving, args.pressure, args.quality, args.out)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
