"""One isolated CPU metadata recheck using the already documented runtime profile.

This source is appended to the frozen finalizer helpers. Original failed
verification files and the frozen binary are never overwritten.
"""


def verify_documented_thread_profile(payload: dict) -> dict:
    approval, manifest = payload['approval'], payload['manifest']
    expected = approval['approved_remote_identity']
    _require((os.getuid(),os.getgid(),socket.gethostname()) ==
             (expected['uid'],expected['gid'],expected['hostname']), 'Remote identity changed')
    base = Path(approval['approved_workspace']); _owned(base)
    root = base/'agent_work/builds/nextgen-native-kv-862c29ca-20261027g'; _owned(root)
    _require(manifest == json.loads((root/'manifest.json').read_text())
             and approval['enabled'] and approval['allow_remote_user_workspace_writes'],
             'Unpinned release repair/no workspace authority')
    expiry=datetime.datetime.fromisoformat(approval['valid_until_utc'].replace('Z','+00:00'))
    _require(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=180)<expiry,
             'CPU verification would exceed current workspace authority')
    binary=root/'artifacts/xllm'; _owned(binary)
    _require(_sha(binary)==manifest['binary_sha256'], 'Frozen release binary changed')
    original=json.loads((root/'verification.json').read_text())
    _require(original['GPU_device_nodes']==[] and original['ldd']['returncode']==0
             and original['help']['returncode']==-6
             and 'Fail to create sampling_thread, Operation not permitted' in original['help']['stderr'],
             'Only the observed CPU seccomp thread-creation failure may be rechecked')
    prepared=json.loads((root/'PREPARED.json').read_text())
    first=json.loads(subprocess.check_output(['docker','inspect',prepared['container_id']],text=True,timeout=15))[0]
    _require(first['State']['Status']=='exited' and first['State']['Pid']==0
             and first['Config']['Labels'].get('kvra.cpu_finalize_id')==manifest['build_id'],
             'Original metadata container is not safely exited')
    revision=root/'verify-v2'
    _require(not revision.exists() and not (root/'release_result.json').exists(),
             'Never repeat or overwrite a verification revision; reconcile read-only')
    revision.mkdir(mode=0o700)
    _json(revision/'INTENT.json',{'operation':'CPU_metadata_only_existing_documented_seccomp_profile',
          'GPU_requested':False,'prior_verification_sha256':_sha(root/'verification.json'),
          'existing_binary_sha256':_sha(binary),'no_recompile_or_test_repeat':True})
    command=json.loads((root/'docker_command.json').read_text())['argv']
    _require(command[:2]==['docker','create'] and approval['approved_image_id'] in command
             and '--privileged' not in command, 'Original metadata container command differs')
    command[command.index('--name')+1]+='-v2'
    index=command.index('--entrypoint')
    command[index:index]=['--security-opt','seccomp=unconfined','--label','kvra.cpu_verify_revision=2',
                         '-e','OPENBLAS_NUM_THREADS=1','-e','OMP_NUM_THREADS=1',
                         '-e','MKL_NUM_THREADS=1','-e','NUMEXPR_NUM_THREADS=1']
    _json(revision/'docker_command.json',{'argv':command})
    identifier=subprocess.check_output(command,text=True,timeout=30).strip()
    _json(revision/'PREPARED.json',{'container_id':identifier,'GPU_requested':False})
    completed=subprocess.run(['docker','start','-a',identifier],capture_output=True,text=True,timeout=110)
    _new(revision/'container.log',(completed.stdout+completed.stderr).encode())
    final=json.loads(subprocess.check_output(['docker','inspect',identifier],text=True,timeout=15))[0]
    _require(final['State']['Status']=='exited' and final['State']['ExitCode']==0
             and not final['State']['OOMKilled'] and not final['HostConfig'].get('Devices')
             and not final['HostConfig'].get('DeviceRequests') and not final['HostConfig']['Privileged'],
             'Metadata verification did not exit cleanly with no GPU')
    value=json.loads(completed.stdout)
    _json(revision/'verification.json',value)
    _require(value['GPU_device_nodes']==[] and value['uid']==os.getuid()
             and value['ldd']['returncode']==0 and 'not found' not in value['ldd']['stdout']+value['ldd']['stderr'],
             'Release dependencies are unresolved')
    text=value['help']['stdout']+value['help']['stderr']
    _require(value['help']['returncode']==0 and all(name in text for name in
             ('kv_cache_backend','kv_cache_k_scale','kv_cache_v_scale','kv_cache_prefill_backend','kv_cache_decode_backend')),
             'Compiled typed native KV options are not exposed by actual help')
    source=base/'agent_work/builds/nextgen-native-kv-862c29ca-20261027f/source_repo'
    _owned(source)
    _require(subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True,timeout=15).strip()
             ==manifest['source_commit'], 'Compiled source changed during metadata repair')
    result={'state':'BINARY_CPU_VERIFIED_FROZEN_NOT_MODEL_OR_GPU_VALIDATED',
            'source_commit':manifest['source_commit'],'source_repo':str(source),'binary':str(binary),
            'binary_sha256':_sha(binary),'binary_bytes':binary.stat().st_size,
            'CXX_tests_passed':manifest['CXX_tests_expected'],'ldd_no_missing_libraries':True,
            'typed_KV_flags_in_actual_binary_help':True,'verification_container_id':identifier,
            'accepted_verification':'verify-v2/verification.json','documented_seccomp_profile':True,
            'GPU_requested':False,'original_F_failure_and_G_first_verification_preserved':True}
    _json(root/'release_result.json',result)
    return result


if __name__=='__main__':
    try:response={'ok':True,**verify_documented_thread_profile(json.load(sys.stdin))}
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as error:
        response={'ok':False,'error':type(error).__name__+': '+str(error)[-4000:]}
    sys.stdout.write(json.dumps(response)+'\n')
