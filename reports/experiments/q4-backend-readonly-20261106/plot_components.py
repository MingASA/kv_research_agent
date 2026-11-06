#!/usr/bin/env python3
"""Plot independently checked MAIN CSVs; no GPU libraries or raw trace execution."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CASES = [('prefill_b1_s8192', 'Prefill: B1, Lq=Lkv=8192'),
         ('decode_b32_s2048', 'Decode: B32, Lq=1, Lkv=2048'),
         ('decode_b32_s16384', 'Decode: B32, Lq=1, Lkv=16384')]
ROUTES = [('mx_prefill_wrapper', 'MX / paged Prefill'),
          ('nv_prefill_wrapper', 'NV / paged Prefill'),
          ('nv_native_decode', 'NV / native Decode')]
COMPONENTS = [('writer_mx', 'Writer MX API'), ('writer_nv', 'Writer NV API'),
              ('attention_mx_prefill_wrapper', 'MX paged Prefill'),
              ('attention_nv_prefill_wrapper', 'NV paged Prefill'),
              ('attention_matched_nv_prefill_wrapper', 'Matched NV Prefill'),
              ('attention_nv_native_decode', 'NV native Decode'),
              ('attention_matched_nv_native_decode', 'Matched NV Decode')]
ORANGE, BLUE = '#ce7b21', '#367ab0'


def read(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def save(fig, output, stem):
    fig.savefig(output / (stem + '.png'), dpi=200, facecolor='white')
    fig.savefig(output / (stem + '.svg'), facecolor='white',
                metadata={'Date': None, 'Creator': 'q4-backend-readonly-20261106'})
    svg = output / (stem + '.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    provenance = json.loads((args.csv_dir / 'READOUT_PROVENANCE.json').read_text())
    assert provenance['production_graph_qualification'] is False
    assert provenance['all_requested_graph_paths_supported'] is False
    assert provenance['main_case_count'] == 3 and provenance['component_count'] == 19
    for name, item in provenance['outputs'].items():
        path = args.csv_dir / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], name
    proxy = read(args.csv_dir / 'component_proxy.csv')
    active = read(args.csv_dir / 'gpu_active.csv')
    spans = read(args.csv_dir / 'event_spans.csv')
    assert len(proxy) == 8 and len(active) == 19
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'q4-20261106',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(3, 2, figsize=(13.3, 9.7), gridspec_kw={'width_ratios': [1.25, 1]})
    for row, (case, title) in enumerate(CASES):
        records = {r['route']: r for r in proxy if r['case'] == case}
        order = [(k, label) for k, label in ROUTES if k in records]
        for col in (0, 1):
            ax = axes[row, col]
            for y, (route, label) in enumerate(order):
                entry = records[route]
                w, a = float(entry['Writer_API_GPU_active_us']), float(entry['Attention_API_GPU_active_us'])
                denominator = float(entry['component_API_GPU_active_proxy_us'])
                assert denominator == w + a
                if col:
                    w, a = 100 * w / denominator, 100 * a / denominator
                ax.barh(y, w, color=ORANGE, height=.56)
                ax.barh(y, a, left=w, color=BLUE, height=.56)
                if col:
                    ax.text(101.2, y, f'{w:.1f}% / {a:.1f}%', va='center', fontsize=9)
                else:
                    ax.text(denominator * 1.015, y, f'{denominator:,.3f}', va='center', fontsize=9)
            ax.set_yticks(range(len(order)), [v[1] for v in order] if not col else [''] * len(order))
            ax.invert_yaxis()
            ax.set_title(title if not col else 'Shares of this API-active proxy', fontsize=11, loc='left')
            ax.set_xlabel('Mean GPU-kernel activity per API call (microseconds)' if not col else 'Percent')
            ax.grid(axis='x', color='#dddddd', linewidth=.5)
            ax.set_axisbelow(True)
            if col:
                ax.set_xlim(0, 128); ax.set_xticks([0, 25, 50, 75, 100])
            else:
                ax.set_xlim(0, max(float(v['component_API_GPU_active_proxy_us']) for v in records.values()) * 1.24)
    fig.suptitle('Production Graph BLOCKED | restricted component profiling\n'
                 'Writer API activity + Attention API activity: two separately verified CUPTI calls each',
                 fontsize=13, y=.985)
    fig.legend(handles=[Patch(color=ORANGE, label='Writer API kernels (includes GPU validation/stack)'),
                        Patch(color=BLUE, label='Attention API kernels (includes compute + merge)')],
               loc='upper center', bbox_to_anchor=(.52, .925), ncol=1, frameon=False, fontsize=9)
    fig.text(.04, .018, 'Pinned FI8bc / H800 SM90 / BF16 / Q24 KV4 D256 / page128. '
             'Kernel-duration sum; not elapsed time, service speed or an internal Unpack split.\n'
             'Two profiler calls support no significance claim. Matched-NV is attention-only. '
             'Exact values and method identities are in the CSVs.', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .87), h_pad=2.3, w_pad=1.7)
    save(fig, args.output, 'component_gpu_active')

    active_map = {(r['case'], r['component'].split('/', 1)[1]): float(r['mean_GPU_active_us']) for r in active}
    span_map = {(r['case'], r['component'], r['mode']): float(r['mean']) * 1000
                for r in spans if r['status'] == 'MEASURED_EVENT_SPAN'}
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.6))
    for ax, (case, title) in zip(axes, CASES):
        order = [(key, label) for key, label in COMPONENTS if (case, key) in active_map]
        for y, (key, label) in enumerate(order):
            ax.scatter(active_map[(case, key)], y - .17, marker='o', s=34, color='#27896d', zorder=3)
            ax.scatter(span_map[(case, key, 'eager')], y, marker='s', s=31, color='#b75c59', zorder=3)
            mode = 'Graph32' if key.startswith('writer_') else 'Graph8'
            if (case, key, mode) in span_map:
                ax.scatter(span_map[(case, key, mode)], y + .17, marker='^', s=37, color='#7965a8', zorder=3)
        ax.set_yticks(range(len(order)), [label for key, label in order], fontsize=9)
        ax.invert_yaxis(); ax.set_xscale('log'); ax.grid(axis='x', which='both', color='#dddddd', linewidth=.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Microseconds per call (log scale)')
    handles = [plt.Line2D([], [], linestyle='', marker='o', color='#27896d', label='CUPTI API GPU-active mean (2 calls)'),
               plt.Line2D([], [], linestyle='', marker='s', color='#b75c59', label='Eager CUDA-event span mean (40 samples)'),
               plt.Line2D([], [], linestyle='', marker='^', color='#7965a8', label='Restricted Graph span mean (40 samples; Writer32/native8)')]
    fig.suptitle('Different timing boundaries | production Graph remains BLOCKED', fontsize=14, y=.985)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.51, .945), ncol=1, frameon=False, fontsize=9)
    fig.text(.025, .025, 'Eager event spans can contain host dispatch/synchronization gaps. '
             'Graph spans are normalized and shown separately; they are never substituted into the CUPTI proxy.\n'
             'Unsafe paged-Prefill Graph is N/A (no marker). Native split attention includes compute and merge. '
             'CUPTI instrumentation and primary timing are separate observations.', fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, .80), w_pad=3)
    save(fig, args.output, 'timing_boundaries')
    receipt = {'schema': 'q4_component_figures_v1', 'production_graph_qualification': False,
               'all_requested_graph_paths_supported': False, 'proxy_rows': len(proxy),
               'component_rows': len(active), 'source_readout': hashlib.sha256(
                   (args.csv_dir / 'READOUT_PROVENANCE.json').read_bytes()).hexdigest(),
               'artifacts': {p.name: {'bytes': p.stat().st_size,
                                     'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                             for p in sorted(args.output.glob('*')) if p.is_file()}}
    (args.output / 'FIGURE_PROVENANCE.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'figure_files': len(receipt['artifacts']), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
