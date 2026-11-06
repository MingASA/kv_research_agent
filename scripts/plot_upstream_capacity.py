"""Plot measured allocator shapes and physical bytes, without throughput inference."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mxfp4-model-sha-manifest', type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    args.out.mkdir(parents=True, exist_ok=False)
    config_path = Path('results/model_tokenizers/Qwen3.8-27B-20261102a/config.json')
    config = json.loads(config_path.read_bytes())['text_config']
    full = config['layer_types'].count('full_attention')
    linear = config['layer_types'].count('linear_attention')
    specifications = [
        ('BF16', 'results/mirrors/upstream-A-graph-e1/files/output/bf16/server.log', 2., None),
        ('FP8', 'results/mirrors/upstream-C-fp8-model-a1/files/output/fp8_e4m3/server.log', 1.,
         'reports/operations/upstream-ac-20261102a/C_FP8_MODEL_INDEPENDENT-a1.json'),
        ('NVFP4', 'results/mirrors/upstream-C-nvfp4-model-a1/files/output/nvfp4/server.log', .5625,
         'reports/operations/upstream-ac-20261102a/C_NVFP4_MODEL_INDEPENDENT-a1.json')]
    rows, sources = [], [config_path]
    mxfp4_runtime = None
    if args.mxfp4_model_sha_manifest is not None:
        from scripts.audit_upstream_run import audit
        proof = audit(args.mxfp4_model_sha_manifest)
        if (proof['kind'] != 'upstream_next_stage_model_qualification'
                or proof['arm'] != 'mxfp4' or proof['original_status'] != 'COMPLETE'):
            raise ValueError('MXFP4 capacity needs actual released model qualification')
        manifest = json.loads(args.mxfp4_model_sha_manifest.read_bytes())
        path = Path(manifest['mirror_directory'])/'files/output/mxfp4/server.log'
        native_lines = [line.split('Native KV backend bound: ', 1)[1]
                        for line in path.read_text().splitlines()
                        if 'Native KV backend bound: ' in line]
        if len(native_lines) != 1:
            raise ValueError('Need one actual MXFP4 native resident layout')
        mxfp4_runtime = json.loads(native_lines[0])
        if (mxfp4_runtime['full_BF16_KV_shadow'] is not False
                or mxfp4_runtime['global_BF16_history_workspace_bytes'] != 0):
            raise ValueError('MXFP4 capacity must be compressed resident storage')
        sources.append(args.mxfp4_model_sha_manifest)
        specifications.append(('MXFP4', str(path), .53125, None))
    for label, name, element_bytes, audit_path in specifications:
        path = Path(name);sources.append(path)
        lines = [s for s in path.read_text().splitlines() if 'kv cache capacity:' in s]
        if len(lines) != 1:
            raise ValueError('Require one actual worker allocation')
        line = lines[0]
        pages = int(re.search(r'blocks: (\d+), slot_size:', line).group(1))
        slots = int(re.search(r'state_slots: (\d+)', line).group(1))
        state = int(re.search(r'linear_slot_size: (\d+)', line).group(1))
        full_bytes = int(pages*128*2*full*config['num_key_value_heads']*config['head_dim']*element_bytes)
        if label == 'MXFP4' and mxfp4_runtime['resident_KV_bytes'] != full_bytes:
            raise ValueError('MXFP4 physical allocation differs from its resident tensor bytes')
        if audit_path:
            audit = Path(audit_path);sources.append(audit)
            if json.loads(audit.read_bytes())['native_runtime'][0]['resident_KV_bytes'] != full_bytes:
                raise ValueError('Actual native byte owners differ from model allocation geometry')
        rows.append({'format': label, 'physical_Full_KV_pages': pages, 'page_tokens': 128,
                     'reserved_padding_pages': 1, 'usable_Full_KV_tokens': (pages-1)*128,
                     'resident_Full_KV_bytes': full_bytes, 'physical_GDN_state_slots': slots,
                     'resident_GDN_state_bytes': slots*state*linear})
    labels = [r['format'] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True)
    bars = axes[0].bar(labels, [r['usable_Full_KV_tokens']/1000 for r in rows],
                       color=['#777777', '#0072B2', '#D55E00', '#009E73'])
    for bar, row in zip(bars, rows):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
                     f"{row['usable_Full_KV_tokens']:,}", ha='center', fontsize=9)
    axes[0].set(ylabel='Usable Full-KV token slots (thousands)', ylim=(0, 710),
                title='Actual allocated capacity (padding excluded)')
    gdn = [r['resident_GDN_state_bytes']/2**30 for r in rows]
    full_gib = [r['resident_Full_KV_bytes']/2**30 for r in rows]
    axes[1].bar(labels, gdn, color='#999999', label='Unquantized GDN states')
    axes[1].bar(labels, full_gib, bottom=gdn, color='#56B4E9', label='Resident Full-KV payload + scales')
    axes[1].set(ylabel='Allocated cache (GiB)', title='Same cache budget; GDN remains unchanged', ylim=(0, 25))
    axes[1].legend(fontsize=8, loc='upper center')
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False);axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True)
    for suffix in ('png', 'svg'):
        fig.savefig(args.out/('capacity.'+suffix), dpi=180)
    svg = args.out/'capacity.svg'
    svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines())+'\n')
    plt.close(fig)
    with (args.out/'capacity.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    (args.out/'provenance.json').open('x').write(json.dumps({
        'classification': 'actual_model_allocator_geometry_and_native_resident_byte_owners',
        'throughput_or_max_admission_capacity_claim': False, 'GDN_bytes_from_actual_slot_size_and_layer_count': True,
        'source_SHA256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}, indent=2)+'\n')
    print(json.dumps({'figure': str(args.out/'capacity.png'), 'rows': rows}))


if __name__ == '__main__':
    main()
