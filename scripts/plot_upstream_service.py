"""Plot complete measured service audits; never turn capacity into throughput."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bf16', 'fp8', 'nvfp4', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    sources = [args.bf16, args.fp8, args.nvfp4]
    audits = [json.loads(p.read_bytes()) for p in sources]
    if [a['arm'] for a in audits] != ['bf16', 'fp8_e4m3', 'nvfp4']:
        raise ValueError('Require the complete three-format service audits')
    names = ['sharegpt-c32', 'sharegpt-c64', 'govreport-c32']
    if any(set(a['cases']) != set(names) for a in audits):
        raise ValueError('Missing original fixed service case')
    args.out.mkdir(parents=True, exist_ok=False)
    labels = ['BF16', 'FP8', 'NVFP4']
    colors = ['#777777', '#0072B2', '#D55E00']
    metrics = [('request_rps', 'Completed requests/s'), ('output_tps', 'Actual output tokens/s'),
               ('ttft_p95_ms', 'P95 TTFT (s)'), ('itl_p95_ms', 'P95 request-mean ITL (ms)')]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    x = np.arange(3)
    rows = []
    for axis, (key, title) in zip(axes.flat, metrics):
        for index, (label, color, audit) in enumerate(zip(labels, colors, audits)):
            scale = .001 if key == 'ttft_p95_ms' else 1
            values = [audit['cases'][name]['profile'][key]*scale for name in names]
            bars = axis.bar(x+(index-1)*.24, values, width=.23, label=label, color=color)
            axis.bar_label(bars, fmt='%.2f' if key == 'request_rps' else '%.1f', fontsize=7, padding=2)
        axis.set_xticks(x, ['ShareGPT C32', 'ShareGPT C64', 'GovReport C32'])
        axis.set(title=title, ylim=(0, axis.get_ylim()[1]*1.16))
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', alpha=.2);axis.set_axisbelow(True)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle('Same H800 / Qwen3.8-27B / fixed requests / natural EOS', fontsize=12)
    for suffix in ('png', 'svg'):
        fig.savefig(args.out/('service.'+suffix), dpi=180)
    plt.close(fig)
    for label, audit in zip(labels, audits):
        for name in names:
            profile = audit['cases'][name]['profile']
            rows.append(dict(format=label, case=name, **{key: profile[key] for key, _ in metrics},
                             actual_output_tokens=profile['actual_output_tokens'],
                             observation_seconds=profile['observation_seconds']))
    with (args.out/'service.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    curve = []
    for label, audit in zip(labels, audits):
        for name in names:
            curve.extend(dict(format=label, case=name, **point)
                         for point in audit['cases'][name]['Goodput']['points'])
    with (args.out/'goodput_all16.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curve[0]), lineterminator='\n')
        writer.writeheader();writer.writerows(curve)
    fig, axes = plt.subplots(3, 1, figsize=(13.8, 7.4), constrained_layout=True)
    for axis, name in zip(axes, names):
        points = audits[0]['cases'][name]['Goodput']['points']
        values = np.array([[point['goodput_rps'] for point in audit['cases'][name]['Goodput']['points']]
                           for audit in audits])
        maximum = max(float(values.max()), .0001)
        heat = axis.imshow(values, aspect='auto', vmin=0, vmax=maximum, cmap='viridis')
        for row in range(3):
            for column in range(16):
                value = values[row, column]
                axis.text(column, row, '0' if value == 0 else f'{value:.3f}', ha='center', va='center',
                          fontsize=7, color='white' if value < .6*maximum else 'black')
        axis.set_yticks(range(3), labels)
        axis.set_xticks(range(16), [f"{p['TTFT_limit_ms']/1000:g}s/{p['request_mean_ITL_limit_ms']:g}ms"
                                   for p in points], rotation=35, ha='right', fontsize=8)
        for boundary in (3.5, 7.5, 11.5):
            axis.axvline(boundary, color='white', linewidth=1)
        axis.set_title(name, fontsize=10)
        fig.colorbar(heat, ax=axis, label='Goodput (requests/s)', fraction=.025, pad=.015)
    axes[-1].set_xlabel('TTFT upper limit / request-mean ITL upper limit')
    fig.suptitle('All 16 predeclared sensitivity points; not a business SLO or semantic-quality gate', fontsize=11)
    for suffix in ('png', 'svg'):
        fig.savefig(args.out/('goodput.'+suffix), dpi=180)
    plt.close(fig)
    for svg in args.out.glob('*.svg'):
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    (args.out/'provenance.json').open('x').write(json.dumps({
        'source_SHA256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'all_original_three_cases_and_formats_shown': True, 'complete_measured_windows': True,
        'independent_repeated_service_samples_per_case': 1,
        'ITL_tail_is_request_mean': True, 'business_SLO_or_quality_accepted': False,
        'Goodput_all_predeclared_threshold_points_exported': 16,
        'Goodput_heatmap_color_scales': 'shared across formats within each case; separate labeled scales per case'}, indent=2)+'\n')
    print(json.dumps({'figure': str(args.out/'service.png'), 'service_rows': len(rows),
                      'Goodput_points_exported': len(curve)}))


if __name__ == '__main__':
    main()
