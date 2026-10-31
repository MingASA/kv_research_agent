#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Apply the pre-export scalar recipe to audited, unlabeled KV range counters."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.controller.durable import write_new_json


def derive(audit_path: Path, rule_path: Path) -> dict:
    audit=json.loads(audit_path.read_text());rule=json.loads(rule_path.read_text())
    if (audit.get('classification')!='confirmed_raw_heldout_BF16_KV_distribution_and_AIPerf_integrity_not_FP8_quality'
            or audit.get('model_quality_certified') is not False or audit.get('new_scales_derived') is not False
            or rule.get('classification')!='proposal_rule_frozen_before_actual_heldout_activation_export_inspection'
            or rule.get('actual_heldout_activation_export_inspected') is not False
            or rule.get('quality_answers_used_to_select_rule') is not False
            or rule.get('baseline_scales')!={'key':.125,'value':.0625}):
        raise ValueError('Require complete independent counter audit and the pre-export unlabeled recipe')
    kinds={};changed=False
    for kind,baseline in rule['baseline_scales'].items():
        rows=[r for r in audit['rows'] if r['kind']==kind]
        if len(rows)!=768 or any(r['nonfinite'] for r in rows):
            raise ValueError('Incomplete/nonfinite activation distribution')
        maximum=max(r['absmax'] for r in rows);clipped=sum(r['clipped'] for r in rows)
        count=sum(r['finite'] for r in rows)
        if not math.isfinite(maximum) or count<1 or (clipped==0)!=(maximum<=448*baseline):
            raise ValueError('Clipping counters and actual range disagree')
        candidate=max(baseline,2.**math.ceil(math.log2(1.1*maximum/448))) if clipped else baseline
        if not math.isfinite(candidate) or candidate<=0 or clipped and candidate<=baseline:
            raise ValueError('Unsupported or inconsistent candidate scale')
        changed|=candidate!=baseline
        kinds[kind]={'baseline_scale':baseline,'candidate_scale':candidate,'finite_values':count,
            'actual_baseline_clipped_values':clipped,'actual_baseline_clipped_pct':100*clipped/count,
            'actual_finite_absmax':maximum,'candidate_coverage_absmax':448*candidate,
            'observed_max_plus_margin_covered':448*candidate>=1.1*maximum if clipped else maximum<=448*candidate}
    return {'classification':'unlabeled_range_derived_global_scale_proposal_not_quality_or_service_result',
        'parent_diagnostic_run_id':audit['run_id'],'source_commit':audit['source_commit'],
        'dataset_sha256':audit['dataset_sha256'],'diagnostic_audit_sha256':hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        'pre_export_rule_sha256':hashlib.sha256(rule_path.read_bytes()).hexdigest(),
        'formula':'candidate=max(baseline,2**ceil(log2(1.1*finite_absmax/448))) iff baseline finite-clipping count>0; otherwise baseline',
        'per_kind':kinds,'scales_changed':changed,'new_GPU_config_justified_by_this_rule':changed,
        'quality_answers_used':False,'model_quality_certified':False,'service_benefit_certified':False,
        'limits':['BF16 heldout short-chat observed writes only; no32K/code/FP8-feedback coverage claim',
            'Larger power2 scale can add underflow and does not improve shared-normal-domain FP8 precision',
            'Any changed config needs actual selected-scale numerical/Graph, fullquality and target service checks']}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('audit','rule','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();result=derive(args.audit,args.rule);write_new_json(args.out,result)
    print(json.dumps(result))


if __name__=='__main__':main()
