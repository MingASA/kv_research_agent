"""Render exact frozen P1 measurements; no inferred service speedups."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE/'components_exact.json'
data = json.loads(SOURCE.read_text())
rows = []
for geometry in data['component_rows']:
    for route in geometry['routes']:
        rows.append(dict(batch=geometry['shape'][0], kv_len=geometry['shape'][1],
                         query_len=geometry['shape'][2], **route))
with (HERE/'components_exact.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
decode = data['component_rows'][1:]
x = np.arange(2)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
colors = ['#356991', '#c27338']
for i, mode in enumerate(['full', 'fast']):
    values = [g['routes'][i]['planning_api_us'] for g in decode]
    bars = axes[0].bar(x+(i-.5)*.34, values, width=.34, color=colors[i],
                       label=['Repaired full planner', 'Candidate native replan'][i])
    axes[0].bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
axes[0].set(xticks=x, xticklabels=['KV 2,048', 'KV 16,384'],
            ylabel='Median host API time (µs / step)', ylim=(0, 380),
            title='Host planning: about 75% lower')
ratios = [g['performance_gate']['graph_event_ratio'] for g in decode]
bars = axes[1].bar(x, ratios, width=.48, color=colors[1])
axes[1].bar_label(bars, labels=[f'{v:.4f}×' for v in ratios], padding=4, fontsize=10)
axes[1].axhline(1., color=colors[0], linewidth=1.2, label='Full planner = 1')
axes[1].axhline(1.05, color='#a13e36', linestyle='--', linewidth=1.2,
                label='Frozen maximum: 1.05')
axes[1].set(xticks=x, xticklabels=['KV 2,048', 'KV 16,384'],
            ylabel='Candidate / full Attention GPU time', ylim=(0, 1.23),
            title='Long-cache case fails the GPU gate')
for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.15)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc='upper left',
              bbox_to_anchor=(0, -.14), ncol=2)
fig.suptitle('MXFP4 planner candidate · H800 / SM90 · B32 / Q24 / KV4 / D256', fontsize=12)
fig.text(.5, .01, '40 GPU / 20 host timing samples, frozen order; not an end-to-end service result.',
         ha='center', fontsize=9, color='#555555')
fig.tight_layout(rect=(0, .11, 1, .94))
for extension in ('png', 'svg'):
    fig.savefig(HERE/f'components.{extension}', dpi=180)
plt.close(fig)
print(json.dumps({'rows': len(rows), 'figures': ['components.png', 'components.svg']}))
