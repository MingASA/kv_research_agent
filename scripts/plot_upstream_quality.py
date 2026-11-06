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
    for name in ('optimized-fp8-pair', 'optimized-fp8-tools-pair', 'mxfp4-pair', 'mxfp4-tools-pair'):
        parser.add_argument('--'+name, type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    paths = [args.fp8_pair, args.nvfp4_pair, args.fp8_tools_pair, args.nvfp4_tools_pair]
    fp8, nv4, ft, nt = [json.loads(p.read_bytes()) for p in paths]
    labels = ['FP8', 'NVFP4'];colors = ['#0072B2', '#D55E00']
    pairs = [fp8, nv4];tools = [ft, nt];expected_arms = ['fp8_e4m3', 'nvfp4']
    for core_path, tools_path, arm, label, color in [
            (args.optimized_fp8_pair, args.optimized_fp8_tools_pair, 'fp8_e4m3', 'FP8 retained', '#CC79A7'),
            (args.mxfp4_pair, args.mxfp4_tools_pair, 'mxfp4', 'MXFP4', '#009E73')]:
        if (core_path is None) != (tools_path is None):
            raise ValueError('Each additional format needs its measured core and tool comparisons')
        if core_path is not None:
            paths.extend([core_path, tools_path])
            pairs.append(json.loads(core_path.read_bytes()));tools.append(json.loads(tools_path.read_bytes()))
            labels.append(label);colors.append(color);expected_arms.append(arm)
    if (not all(d['same_frozen_official_base_A_C'] for d in pairs+tools)
            or any(p['reference_run'] != fp8['reference_run'] for p in pairs)
            or any(t['reference_run_id'] != ft['reference_run_id'] for t in tools)
            or any(t['BF16_correct'] != ft['BF16_correct'] for t in tools)
            or [d['candidate_arm'] for d in pairs] != expected_arms
            or [d['candidate_arm'] for d in tools] != expected_arms):
        raise ValueError('Require complete matched upstream A/C quality and tool comparisons')
    args.out.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), constrained_layout=True)
    rows = []
    for axis, (metric, task_name) in zip(axes[:3], [
            ('GSM8K flexible EM', 'gsm8k'), ('LongBench core macro', None),
            ('RepoBench-P', 'repobench-p')]):
        for y, (label, color, pair) in enumerate(zip(labels, colors, pairs)):
            if task_name is not None:
                task = pair['task_differences'][task_name]
                delta = task['delta_percentage_points'];lo, hi = task['paired_95pct_context_cluster_CI_pp']
            else:
                delta = pair['LongBench_core_macro_delta_pp']
                lo, hi = pair['LongBench_macro_paired_95pct_context_cluster_CI_pp']
            axis.plot([lo, hi], [y, y], color=color, linewidth=2)
            axis.scatter([delta], [y], color=color, s=35, zorder=3)
            rows.append({'metric': metric, 'format': label, 'delta_pp': delta, 'CI95_low_pp': lo, 'CI95_high_pp': hi})
        axis.axvline(0, color='#777777', linestyle='--', linewidth=1)
        axis.set(yticks=list(range(len(pairs))), yticklabels=labels, ylim=(-.6, len(pairs)-.4),
                 xlabel='C minus official BF16 (percentage points)', title=metric)
        axis.grid(axis='x', alpha=.2)
    counts = [ft['BF16_correct']]+[tool['candidate_correct'] for tool in tools]
    bars = axes[3].bar(['BF16']+[label.replace(' retained', '\nretained') for label in labels],
                       counts, color=['#777777']+colors)
    axes[3].bar_label(bars, labels=[f'{n}/32' for n in counts], padding=3, fontsize=8)
    axes[3].set(ylabel='Correct original tasks', ylim=(0, 35), title='Native Tool-Agent (all 32 tasks)')
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
        'tool_counts': dict(zip(['BF16']+labels, counts)),
        'noninferiority_or_production_quality_accepted': False}, indent=2)+'\n')
    print(json.dumps({'figure': str(args.out/'quality.png'), 'quality_delta_rows': len(rows)}))


if __name__ == '__main__':
    main()
