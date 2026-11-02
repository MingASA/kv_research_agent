"""Plot paired measured quality changes with their original uncertainty units."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fp8-pair', 'nvfp4-pair', 'fp8-tools-pair', 'nvfp4-tools-pair', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    paths = [args.fp8_pair, args.nvfp4_pair, args.fp8_tools_pair, args.nvfp4_tools_pair]
    fp8, nv4, ft, nt = [json.loads(p.read_bytes()) for p in paths]
    if (not all(d['same_frozen_official_base_A_C'] for d in (fp8, nv4, ft, nt))
            or fp8['reference_run'] != nv4['reference_run']
            or ft['reference_run_id'] != nt['reference_run_id']
            or ft['BF16_correct'] != nt['BF16_correct']
            or [d['candidate_arm'] for d in (fp8, nv4, ft, nt)] != ['fp8_e4m3', 'nvfp4']*2):
        raise ValueError('Require complete matched upstream A/C quality and tool comparisons')
    args.out.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    colors = ['#0072B2', '#D55E00']
    rows = []
    for axis, metric in zip(axes[:2], ['GSM8K flexible EM', 'LongBench core macro']):
        for y, (label, color, pair) in enumerate(zip(['FP8', 'NVFP4'], colors, [fp8, nv4])):
            if metric.startswith('GSM'):
                task = pair['task_differences']['gsm8k']
                delta = task['delta_percentage_points'];lo, hi = task['paired_95pct_context_cluster_CI_pp']
            else:
                delta = pair['LongBench_core_macro_delta_pp']
                lo, hi = pair['LongBench_macro_paired_95pct_context_cluster_CI_pp']
            axis.plot([lo, hi], [y, y], color=color, linewidth=2)
            axis.scatter([delta], [y], color=color, s=35, zorder=3)
            rows.append({'metric': metric, 'format': label, 'delta_pp': delta, 'CI95_low_pp': lo, 'CI95_high_pp': hi})
        axis.axvline(0, color='#777777', linestyle='--', linewidth=1)
        axis.set(yticks=[0, 1], yticklabels=['FP8', 'NVFP4'], ylim=(-.6, 1.6),
                 xlabel='C minus official BF16 (percentage points)', title=metric)
        axis.grid(axis='x', alpha=.2)
    counts = [ft['BF16_correct'], ft['candidate_correct'], nt['candidate_correct']]
    bars = axes[2].bar(['BF16', 'FP8', 'NVFP4'], counts, color=['#777777']+colors)
    axes[2].bar_label(bars, labels=[f'{n}/32' for n in counts], padding=3)
    axes[2].set(ylabel='Correct original tasks', ylim=(0, 35), title='Native Tool-Agent (all 32 tasks)')
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Paired quality: 95% bootstrap intervals; tool scores are descriptive', fontsize=11)
    for suffix in ('png', 'svg'):
        fig.savefig(args.out/('quality.'+suffix), dpi=180)
    plt.close(fig)
    svg = args.out/'quality.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    with (args.out/'quality_delta.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    (args.out/'provenance.json').open('x').write(json.dumps({
        'source_SHA256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'GSM_unit': 'paired questions', 'LongBench_unit': 'complete same-context question clusters within task',
        'LongBench_task_macro_weights': 'equal eight fixed core tasks; one supplement excluded',
        'tool_counts': dict(zip(['BF16', 'FP8', 'NVFP4'], counts)),
        'noninferiority_or_production_quality_accepted': False}, indent=2)+'\n')
    print(json.dumps({'figure': str(args.out/'quality.png'), 'quality_delta_rows': len(rows)}))


if __name__ == '__main__':
    main()
