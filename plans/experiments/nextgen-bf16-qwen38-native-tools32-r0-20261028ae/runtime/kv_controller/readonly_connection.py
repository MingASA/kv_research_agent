"""A private ten-minute SSH master for read-only recovery traffic only.

Host/user/pin/forwarding settings come from PinnedSSH. No server settings are
modified. Mutations use the ordinary independent dispatch path.
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path
import stat
import subprocess
import time

from .durable import secure_directory,write_new_json
from .processes import process_identity
from .transport import PinnedSSH


class ReadOnlyConnection:
    def __init__(self,approval: dict,directory: Path) -> None:
        self.directory=directory;secure_directory(directory)
        self.socket=directory/'mux'
        if len(str(self.socket).encode())>100:raise ValueError('Private SSH control socket path is too long')
        self.transport=PinnedSSH(approval,directory)

    def start(self) -> dict:
        if (self.directory/'MASTER_INTENT.json').exists() or self.socket.exists():
            raise ValueError('Existing read-only master must be inspected,never started twice')
        args=list(self.transport.argv)
        args=[('ControlMaster=yes' if value=='ControlMaster=no' else
               'ControlPath='+str(self.socket) if value=='ControlPath=none' else value) for value in args]
        args[-1:-1]=['-M','-N']
        argv=['/usr/bin/timeout','--kill-after=5s','600s',*args]
        write_new_json(self.directory/'MASTER_INTENT.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'argv':argv,'max_seconds':600,'remote_command_or_mutation':False})
        with (self.directory/'master.stdout.log').open('xb') as stdout,(self.directory/'master.stderr.log').open('xb') as stderr:
            process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
        write_new_json(self.directory/'MASTER_PROCESS.json',{'identity':process_identity(process.pid),'argv':argv})
        until=time.monotonic()+25
        while process.poll() is None and time.monotonic()<until:
            if self.socket.exists():
                info=self.socket.lstat()
                if not stat.S_ISSOCK(info.st_mode) or info.st_uid!=os.getuid():
                    raise ValueError('Unexpected read-only SSH socket ownership/type')
                result={'ready':True,'socket':str(self.socket),'local_wrapper_pid':process.pid,
                        'remote_command_or_mutation':False,'max_seconds':600}
                write_new_json(self.directory/'MASTER_READY.json',result);return result
            time.sleep(.1)
        result={'ready':False,'local_wrapper_pid':process.pid,'process_exit_code':process.poll(),
                'remote_command_or_mutation':False,'max_seconds':600,
                'stderr_tail':(self.directory/'master.stderr.log').read_text(errors='replace')[-1000:]}
        write_new_json(self.directory/'MASTER_NOT_READY.json',result);return result

    def rpc(self,source: str,payload: dict,*,mutation: bool = False,timeout: int = 60) -> dict:
        if mutation:raise ValueError('The shared recovery connection is read-only;use normal unique dispatch')
        if not self.socket.exists() or not (self.directory/'MASTER_READY.json').is_file():
            raise ValueError('Read-only master is not available;use bounded recovery after cooldown')
        original=self.transport.argv
        self.transport.argv=['ControlPath='+str(self.socket) if value=='ControlPath=none' else value for value in original]
        try:return self.transport.rpc(source,payload,mutation=False,timeout=timeout)
        finally:self.transport.argv=original
