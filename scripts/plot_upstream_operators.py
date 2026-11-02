"""Plot actual native/direct SM90 operator evidence, without service claims."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    args.out.mkdir(parents=True, exist_ok=False)
    operators = Path('results/mirrors/upstream-C-operators-a1/files/output/fp8_e4m3/operators')
    diagnosis = Path('results/mirrors/upstream-C-fp8-direct-b1/files/output/fp8_e4m3/diagnosis/diagnosis.json')
    diagnostic = json.loads(diagnosis.read_bytes())
    sources = [diagnosis]
    values = [('FP8 FA3 / XQA', '#0072B2', diagnostic['prefill_timings'], diagnostic['timings'])]
    for name, label, color in [('fp8_e4m3-fa2-flashinfer', 'FP8 FA2 / FlashInfer', '#009E73'),
                               ('nvfp4-fa2-flashinfer', 'NVFP4 FA2 / FlashInfer', '#D55E00')]:
        paths = [operators/name/n for n in ('prefill_timings.json', 'timings.json')]
        sources.extend(paths)
        values.append((label, color, *[json.loads(p.read_bytes()) for p in paths]))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    rows = []
    for label, color, prefill, decode in values:
        for axis, phase, records, xname, native, direct in [
            (axes[0], 'prefill', prefill, 'context', 'native_prefill_writer_query_attention_us', 'direct_FlashInfer_attention_us'),
            (axes[1], 'decode-L8192', [r for r in decode if r['context'] == 8192], 'batch',
             'native_decode_writer_attention_graph_us', 'direct_FlashInfer_attention_graph_us')]:
            axis.plot([r[xname] for r in records], [r[native] for r in records], 'o-', color=color, label=label)
            axis.plot([r[xname] for r in records], [r[direct] for r in records], 'x--', color=color, alpha=.75)
            rows.extend({'route': label, 'phase': phase, 'batch': r['batch'], 'context': r['context'],
                         'native_us': r[native], 'direct_attention_us': r[direct]} for r in records)
    axes[0].set(xscale='log', yscale='log', xlabel='Context tokens (B=1)', ylabel='Device time (microseconds)',
                title='Prefill: Writer + Q conversion + Attention')
    axes[1].set(yscale='log', xlabel='Batch (context=8192)', title='Decode: Writer + Attention')
    axes[1].set_xticks([1, 16, 32])
    for axis in axes:
        axis.grid(alpha=.2, which='both')
        axis.spines[['top', 'right']].set_visible(False)
    axes[0].legend(fontsize=8, loc='upper left')
    fig.suptitle('H800 / SM90, Q24-KV4-D256: solid = integrated, dashed = direct library', fontsize=11)
    for suffix in ('png', 'svg'):
        fig.savefig(args.out/('operators.'+suffix), dpi=180)
    svg = args.out/'operators.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)
    with (args.out/'operators.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    (args.out/'provenance.json').open('x').write(json.dumps({
        'classification': 'actual_GPU_operator_measurements_not_service_performance',
        'iterations': 50, 'planning_excluded_from_device_timings': True,
        'same_original_dense_FA3_failure_preserved': True,
        'source_SHA256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'prefill_production_Graph_coverage_claim': False,
        'source_commit': 'da7c6446ff4361c8060633b04d6bfb77e6f737ff'}, indent=2)+'\n')
    print(json.dumps({'figure': str(args.out/'operators.png'), 'rows': len(rows)}))


if __name__ == '__main__':
    main()
