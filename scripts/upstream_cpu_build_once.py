"""Prepare and start one new, GPU-invisible CPU build with exclusive receipts."""

import argparse
import base64
import datetime
import json
from pathlib import Path

from scripts.controller.transport import PinnedSSH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    directory = args.plan.resolve()
    manifest = json.loads((directory/'manifest.json').read_bytes())
    approval = json.loads((Path.home()/'.config/kv-research/approval.json').read_bytes())
    ssh = PinnedSSH(approval, directory/'pin')
    source = (directory/'build_rpc.py').read_text()
    payload = {'approval': approval, 'manifest': manifest, 'operation': 'inspect'}
    inspected = ssh.rpc(source, payload)
    with (directory/'LOCAL_INSPECT.json').open('x') as f:
        json.dump(inspected, f, indent=2)
    for operation in ('prepare', 'start'):
        with (directory/('LOCAL_'+operation.upper()+'_INTENT.json')).open('x') as f:
            json.dump({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       'operation': operation, 'do_not_repeat': True}, f)
        payload['operation'] = operation
        if operation == 'prepare':
            payload['assets'] = {name: {'sha256': digest,
                'base64': base64.b64encode((directory/name).read_bytes()).decode()}
                for name, digest in manifest['assets_sha256'].items()}
        result = ssh.rpc(source, payload, mutation=True, timeout=600)
        with (directory/('LOCAL_'+operation.upper()+'_ACK.json')).open('x') as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
