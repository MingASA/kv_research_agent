"""Compare paired core scores with dataset-appropriate uncertainty units."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.audit_nextgen_validation import _jsonl
from scripts.controller.durable import write_new_json


def _cluster_bootstrap(differences: np.ndarray, groups: list[str], rng: np.random.Generator, repeats: int) -> np.ndarray:
    """Resample complete context clusters, retaining their question weights."""
    unique=sorted(set(groups))
    values=[differences[np.asarray(groups)==group] for group in unique]
    totals=np.array([float(value.sum()) for value in values])
    counts=np.array([len(value) for value in values])
    sampled=rng.integers(0,len(unique),(repeats,len(unique)))
    return totals[sampled].sum(axis=1)/counts[sampled].sum(axis=1)


def compare(reference: Path, candidate: Path, *, repeats: int = 4000, upstream: bool = False,
            reference_manifest: Path | None = None, candidate_manifest: Path | None = None,
            native_reference: str | None = None) -> dict:
    if native_reference is not None and not upstream:
        raise ValueError('A previous-format reference needs original upstream raw manifests')
    baseline=json.loads((reference/'summary.json').read_text())
    low=json.loads((candidate/'summary.json').read_text())
    common = ('model_id','canonical_API_payloads_sha256','labels_sha256','scoring_protocol_sha256','service')
    deployments = None
    if upstream:
        from scripts.audit_upstream_run import audit as audit_upstream
        from scripts.controller.upstream_qualification import BASELINE_COMMIT
        if reference_manifest is None or candidate_manifest is None:
            raise ValueError('Upstream A/C quality comparison requires both original SHA mirrors')
        requests = []
        for summary, path in ((baseline, reference_manifest), (low, candidate_manifest)):
            lifecycle = audit_upstream(path)
            manifest = json.loads(path.read_bytes())
            request = json.loads((Path(manifest['mirror_directory'])/'files/ticket.json').read_bytes())['request']
            if (lifecycle['run_id'] != summary['run_id'] or summary.get('same_version_upstream_A_C') is not True
                    or request['workload']['kind'] != 'upstream_AC_frozen_core_quality1720'
                    or request['server_source_commit'] != summary['server_source_commit']
                    or request['binary_sha256'] != summary['binary_sha256']
                    or request.get('native_configuration') != summary.get('native_configuration')):
                raise ValueError('Quality summary differs from its original actual upstream run')
            requests.append(request)
        a, c = requests
        for field in ('model_id','model_config_sha256','image_id','gpu_uuids','service','upstream_commit'):
            if a[field] != c[field]:
                raise ValueError('Paired upstream model/image/GPU/service differs: '+field)
        from scripts.controller.next_stage_deployment import validate_pair_sources
        validate_pair_sources(a,c,native_reference)
        deployments = {'reference_source': a['server_source_commit'], 'candidate_source': c['server_source_commit'],
                       'reference_python_source':a['python_source_commit'],'candidate_python_source':c['python_source_commit'],
                       'reference_format':a['arms'][0],'candidate_format':c['arms'][0],
                       'official_base': a['upstream_commit'], 'common_buildfix_base': BASELINE_COMMIT,
                       'same_physical_GPU': a['gpu_uuids'], 'same_image': a['image_id']}
    else:
        common += ('binary_sha256','server_source_commit')
    for field in common:
        if baseline[field]!=low[field]:
            raise ValueError('Paired quality configurations/input identities differ: '+field)
    left_rows=_jsonl(reference/'per_record_scores.jsonl');right_rows=_jsonl(candidate/'per_record_scores.jsonl')
    left={row['id']:row for row in left_rows};right={row['id']:row for row in right_rows}
    if set(left)!=set(right) or len(left)!=1720 or len(left_rows)!=1720 or len(right_rows)!=1720:
        raise ValueError('Missing/duplicate paired quality record')
    rng=np.random.default_rng(20261027)
    tasks={};longbench_boot=[]
    for task in sorted({row['task'] for row in left.values() if row['group']=='core'}):
        ids=sorted(identifier for identifier,row in left.items() if row['task']==task and row['group']=='core')
        for identifier in ids:
            if any(left[identifier][field]!=right[identifier][field] for field in ('task','group','source_index','context_sha256')):
                raise ValueError('A paired record maps to another source context')
        differences=np.array([right[identifier]['score']-left[identifier]['score'] for identifier in ids])
        groups=[left[identifier]['context_sha256'] or identifier for identifier in ids]
        boot=_cluster_bootstrap(differences,groups,rng,repeats)
        if task!='gsm8k':longbench_boot.append(boot)
        tasks[task]={'paired_records':len(ids),'independent_context_clusters':len(set(groups)),
                     'delta_percentage_points':float(differences.mean()*100),
                     'paired_95pct_context_cluster_CI_pp':(np.quantile(boot,[.025,.975])*100).tolist(),
                     'improved_questions':int((differences>0).sum()),'worsened_questions':int((differences<0).sum()),
                     'ties':int((differences==0).sum()),
                     'reference_capped':sum(left[identifier]['finish_reason']=='length' for identifier in ids),
                     'candidate_capped':sum(right[identifier]['finish_reason']=='length' for identifier in ids)}
    macro=np.stack(longbench_boot).mean(axis=0)*100
    return {'schema':'nextgen_paired_core_quality_comparison_v1','reference_run':baseline['run_id'],
            'same_frozen_official_base_A_C':upstream and native_reference is None,
            'same_frozen_official_base':upstream,'upstream_deployments':deployments,
            'candidate_run':low['run_id'],'reference_arm':baseline['arm'],'candidate_arm':low['arm'],
            'reference_summary_sha256':hashlib.sha256((reference/'summary.json').read_bytes()).hexdigest(),
            'candidate_summary_sha256':hashlib.sha256((candidate/'summary.json').read_bytes()).hexdigest(),
            'task_differences':tasks,'LongBench_core_macro_delta_pp':
                 low['LongBench_core_macro_percent_unrounded']-baseline['LongBench_core_macro_percent_unrounded'],
            'LongBench_macro_paired_95pct_context_cluster_CI_pp':np.quantile(macro,[.025,.975]).tolist(),
            'bootstrap_repeats':repeats,'bootstrap_seed':20261027,
            'bootstrap_scope':'Paired questions for GSM; complete same-context question clusters within each fixed LongBench task. '
                              'Eight core tasks retain equal macro weight; one long supplement excluded.',
            'user_quality_tolerance_provided':False,'production_quality_accepted':False,
            'service_performance_claim':False}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--upstream',action='store_true')
    parser.add_argument('--reference-sha-manifest',type=Path)
    parser.add_argument('--candidate-sha-manifest',type=Path)
    parser.add_argument('--native-reference',choices=['fp8_e4m3','nvfp4'])
    args=parser.parse_args();write_new_json(args.out,compare(args.reference,args.candidate,upstream=args.upstream,
        reference_manifest=args.reference_sha_manifest,candidate_manifest=args.candidate_sha_manifest,
        native_reference=args.native_reference))


if __name__=='__main__':main()
