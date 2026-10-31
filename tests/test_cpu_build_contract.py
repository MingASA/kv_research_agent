"""CPU-only build admission and private-output checks, without SSH/Docker."""

import datetime
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.native_cpu_build_driver import _run, _tree_bytes
from scripts.native_cpu_build_rpc import _owned, _shared_headroom, run


class CpuBuildContractTests(unittest.TestCase):
    def test_high_load_average_does_not_replace_measured_CPU_availability(self):
        for busy,idle,expected in [(37,63,True),(90,10,False)]:
            samples=iter([f'cpu 100 0 0 100 0 0 0 0\n',f'cpu {100+busy} 0 0 {100+idle} 0 0 0 0\n'])
            def read(path,*_args,**_kwargs):
                name=str(path)
                if name=='/proc/meminfo':return 'MemAvailable: 1073741824 kB\n'
                if name=='/proc/stat':return next(samples)
                if name.startswith('/proc/pressure/'):
                    return 'some avg10=0.00 avg60=0.00 avg300=0.00 total=0\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0\n'
                raise ValueError('Unexpected resource read: '+name)
            with self.subTest(busy=busy),patch.object(Path,'read_text',read),patch('time.sleep'), \
                 patch('os.cpu_count',return_value=192),patch('os.getloadavg',return_value=(150,150,150)):
                value=_shared_headroom()
                self.assertEqual(value['can_prepare'],expected)
                self.assertAlmostEqual(value['CPU_idle_fraction'],idle/100)

    def test_accounting_does_not_follow_borrowed_readonly_dependency_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary);root=base/'private';root.mkdir()
            (base/'borrowed').mkdir();(base/'borrowed/large').write_bytes(b'x'*100)
            (root/'registry').symlink_to(base/'borrowed',target_is_directory=True)
            (root/'own').write_bytes(b'123')
            self.assertEqual(_tree_bytes(root),3)

    def test_noncanonical_dependency_is_not_adopted_as_writable_owned_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary);(base/'real').mkdir();(base/'alias').symlink_to(base/'real',target_is_directory=True)
            with self.assertRaises(ValueError):_owned(base/'alias')

    def test_deadline_stops_only_the_newly_created_child_group_and_preserves_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with self.assertRaises(TimeoutError):
                _run([sys.executable,'-c','import time;time.sleep(100)'],root,'owned_timeout',dict(os.environ),0)
            self.assertTrue((root/'owned_timeout.log').is_file())
            self.assertTrue((root/'owned_timeout_command.json').is_file())

    def test_inspection_is_readonly_and_requires_shared_resource_headroom(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary);source=base/'old_source';source.mkdir()
            build_id='nextgen-native-kv-12345678-20261027a'
            image='sha256:'+'a'*64
            approval={'approved_remote_identity':{'uid':os.getuid(),'gid':os.getgid(),'hostname':socket.gethostname()},
                      'approved_workspace':str(base),'approved_image_id':image,'enabled':True,
                      'allow_remote_user_workspace_writes':True,'execution_contract':'kv-controller-v1',
                      'valid_until_utc':(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(days=1)).isoformat()}
            manifest={'build_id':build_id,'host_root':str(base/'agent_work/builds'/build_id),'uid':os.getuid(),
                      'image_id':image,'source_commit':'a'*40,'baseline_source':str(source)}
            original=Path.read_text
            def read(path,*args,**kwargs):
                name=str(path)
                if name=='/proc/meminfo':return 'MemAvailable: 1073741824 kB\n'
                if name=='/proc/stat':return next(samples)
                if name.startswith('/proc/pressure/'):
                    return 'some avg10=0.00 avg60=0.00 avg300=0.00 total=0\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0\n'
                return original(path,*args,**kwargs)
            samples=iter(['cpu 100 0 0 100 0 0 0 0\n','cpu 110 0 0 190 0 0 0 0\n']*2)
            with patch.object(Path,'read_text',read),patch('os.getloadavg',return_value=(0,0,0)), \
                 patch('time.sleep'),patch('os.cpu_count',return_value=4), \
                 patch('subprocess.check_output',return_value=json.dumps([{'Id':image}])) as commands:
                result=run({'operation':'inspect','approval':approval,'manifest':manifest})
                self.assertTrue(result['read_only'])
                self.assertFalse(result['GPU_requested'])
                commands.assert_called_once_with(['docker','image','inspect',image],text=True,timeout=15)
                self.assertFalse((base/'agent_work').exists())
                with patch('os.getloadavg',return_value=(1e9,1e9,1e9)),self.assertRaises(ValueError):
                    run({'operation':'inspect','approval':approval,'manifest':manifest})


if __name__=='__main__':unittest.main()
