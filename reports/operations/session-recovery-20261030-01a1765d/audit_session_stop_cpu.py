"""Independently audit the actual CPU CLI wire records; no GPU or model calls."""
import collections
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent
DATA=BASE.parents[2]/'results/prepared/nextgen-production-real-inputs-20261028a'


def load(path):return json.loads(path.read_bytes())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def encoded(value):return json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
def records(path):return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def main():
    positive=BASE/'session-stop-CPU-positive-a1'
    negative=BASE/'session-stop-CPU-negative-a1'
    assert load(positive/'EXIT.json')['exit_code']==0
    assert load(negative/'EXIT.json')['exit_code']==1
    control=load(negative/'full_cli_cases/long_prefix16x2/proof.json')
    assert control['process_return_code']==0 and control['actual_HTTP_requests']==32
    assert control['all_actual_payload_counts_match'] is False and control['session_limit_enabled'] is False
    summary=load(positive/'summary.json');manifest=load(DATA/'manifest.json')
    names={'warmup_chat16.json','chat128.json','warmup_long2.json','long_unique16.json','long_prefix16x2.json'}
    assert {Path(report['input']).name for report in summary['reports']}==names
    reports=[];total=0
    for report in summary['reports']:
        source=Path(report['input']);name=source.name;directory=positive/'full_cli_cases'/source.stem
        expected={(row['session_id'],turn):payload for row in load(source)['data']
                  for turn,payload in enumerate(row['payloads'])}
        assert sha(source)==manifest['files'][name]['sha256']==report['input_sha256']
        assert report['process_return_code']==0 and report['all_actual_payload_counts_match'] is True
        command=report['command'];argc=lambda flag:command[command.index(flag)+1]
        assert int(argc('--num-sessions'))==manifest['files'][name]['sessions']
        assert int(argc('--request-count'))==len(expected)==report['actual_HTTP_requests']
        assert int(argc('--concurrency'))==min(16,manifest['files'][name]['sessions'])
        assert report['session_limit_enabled'] is True and report['heterogeneous_latency'] is True
        assert not any('warmup' in item or item=='--profile' for item in command if item.startswith('--'))
        wanted=collections.Counter(hashlib.sha256(encoded(payload)).hexdigest() for payload in expected.values())
        for field in ('payload_sha256','wire_sha256'):
            assert collections.Counter(row[field] for row in report['wire_observations'])==wanted
        raw=records(directory/'artifacts/profile_export_raw.jsonl')
        metric=records(directory/'artifacts/profile_export.jsonl')
        assert len(raw)==len(metric)==len(expected)
        ids=set();turns=set();times={}
        for row in raw:
            meta=row['metadata'];key=(meta['conversation_id'],meta['turn_index']);identifier=meta['x_request_id']
            assert key in expected and key not in turns and identifier not in ids
            assert meta['benchmark_phase']=='profiling' and meta['was_cancelled'] is False
            assert meta['context_overflow_skip'] is False and row['status']==200
            assert row['payload']==expected[key]
            payload=row['payload'];assert payload['stream'] is True and payload['chat_template_kwargs']=={'enable_thinking':False}
            assert not payload.get('ignore_eos',False) and payload.get('stop') is None
            packets=[packet['value'] for response in row['responses'] for packet in response['packets'] if packet['name']=='data']
            assert packets[-1]=='[DONE]'
            last=json.loads(packets[-2]);assert last['choices'][0]['finish_reason']=='stop'
            assert last['usage']=={'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}
            assert meta['request_start_ns']<meta['request_end_ns']
            ids.add(identifier);turns.add(key);times[key]=(meta['request_start_ns'],meta['request_end_ns'])
        assert turns==set(expected) and ids=={row['metadata']['x_request_id'] for row in metric}
        for session,turn in turns:
            if turn:assert times[(session,turn-1)][1]<=times[(session,turn)][0]
        for row in metric:
            assert row.get('error') is None and row['metadata']['benchmark_phase']=='profiling'
        exported=load(directory/'artifacts/profile_export_aiperf.json')
        assert exported['is_complete'] is True and exported['was_cancelled'] is False and not exported.get('error_summary')
        reports.append({'input':name,'input_sha256':sha(source),'sessions':manifest['files'][name]['sessions'],
            'actual_wire_requests':len(raw),'actual_unique_session_turns':len(turns),
            'exact_payload_wire_counts_and_session_order_verified':True,
            'no_embedded_warmup_or_forced_stops':True,'CPU_fixture_SSE_status_usage_verified':True,
            'CLI_command':command,'raw_sha256':sha(directory/'artifacts/profile_export_raw.jsonl'),
            'metric_sha256':sha(directory/'artifacts/profile_export.jsonl')})
        total+=len(raw)
    assert total==194
    sdk=load(BASE/'session-stop-SDK-fresh-readonly-a1/COMPARISON.json')
    assert sdk['all_AIPerf_Python_source_files_match_fresh_remote'] is True
    result={'schema':'nextgen-full-cli-session-stop-v2','all_checks_pass':True,
        'AIPerf_version':'0.13.0','observed_wire_request_total':total,
        'all_cases_session_limit_verified':True,'heterogeneous_latency_verified':True,
        'multi_turn_concurrency':16,'request_count_only_negative_control_reproduced':True,
        'CPU_fixture_only':True,'GPU_used':False,'model_inference':False,'performance_measured':False,
        'reports':reports,'actual_full_CLI_summary_sha256':sha(positive/'summary.json'),
        'negative_control_proof_sha256':sha(negative/'full_cli_cases/long_prefix16x2/proof.json'),
        'remote_SDK_identity':sdk,'auditor_source_sha256':sha(Path(__file__)),
        'scope':'Actual full installed CLI, five unchanged frozen inputs, all194 exact HTTP payloads and original session turns. Fake one-token CPU responses with deliberate fast/slow latency; no quality or performance conclusion.'}
    destination=BASE/'session-stop-full-CLI-independent-CPU-proof-a1.json'
    with destination.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({'all_checks_pass':True,'actual_wire_requests':total,'proof':str(destination),'sha256':sha(destination)}))


if __name__=='__main__':main()
