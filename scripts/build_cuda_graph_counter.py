"""Build a diagnostic observer and verify transparent forwarding without CUDA."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile


def build(output: Path):
    output.mkdir()
    source = Path(__file__).with_name('cuda_graph_counter.c')
    library = output/'libkvra_graph_counter.so'
    command = ['cc', '-O2', '-Wall', '-Wextra', '-Werror', '-shared', '-fPIC',
               str(source), '-o', str(library), '-ldl', '-pthread']
    subprocess.run(command, check=True, capture_output=True)
    with tempfile.TemporaryDirectory(prefix='kvra-counter-forwarding-') as directory:
        root = Path(directory)
        fake = root/'reference.c'
        fake.write_text('''#include <stdint.h>
int cudaGraphLaunch(void *g, void *s) { return g==(void*)3 && s==(void*)4 ? 0 : 17; }
int cudaGraphLaunch_ptsz(void *g, void *s) { return g==(void*)5 && s==(void*)6 ? 0 : 19; }
''')
        subprocess.run(['cc', '-shared', '-fPIC', str(fake), '-o', str(root/'libreference.so')], check=True)
        probe = root/'probe.c'
        probe.write_text('''extern int cudaGraphLaunch(void*,void*);
extern int cudaGraphLaunch_ptsz(void*,void*);
int main(void) { return !(cudaGraphLaunch((void*)3,(void*)4)==0 &&
 cudaGraphLaunch((void*)1,(void*)2)==17 && cudaGraphLaunch_ptsz((void*)5,(void*)6)==0 &&
 cudaGraphLaunch_ptsz((void*)1,(void*)2)==19); }
''')
        subprocess.run(['cc', str(probe), '-L'+str(root), '-lreference', '-Wl,-rpath,'+str(root),
                        '-o', str(root/'probe')], check=True)
        counter = root/'counter.bin'
        subprocess.run([str(root/'probe')], env=dict(os.environ, LD_PRELOAD=str(library.resolve()),
                       KVRA_GRAPH_COUNTER_FILE=str(counter)), check=True)
        values = struct.unpack('=8Q', counter.read_bytes())
        assert values[0:2] == (0x4b56524147524150, 1) and values[3:7] == (4, 1, 1, 2)
        subprocess.run(['cc', '-shared', '-fPIC', str(fake), '-o', str(root/'libcudart.so.13')], check=True)
        local_probe = root/'local_probe.c'
        local_probe.write_text('''#include <dlfcn.h>
typedef int (*fn)(void*,void*);
int main(void) { if(!dlopen("libcudart.so.13",RTLD_NOW|RTLD_LOCAL)) return 2;
 fn a=(fn)dlsym(RTLD_DEFAULT,"cudaGraphLaunch");
 fn b=(fn)dlsym(RTLD_DEFAULT,"cudaGraphLaunch_ptsz");
 return !(a((void*)3,(void*)4)==0 && a((void*)1,(void*)2)==17 &&
 b((void*)5,(void*)6)==0 && b((void*)1,(void*)2)==19); }
''')
        subprocess.run(['cc', '-D_GNU_SOURCE', str(local_probe), '-ldl', '-o', str(root/'local_probe')], check=True)
        second = root/'local_counter.bin'
        subprocess.run([str(root/'local_probe')], check=True,
            env=dict(os.environ, LD_PRELOAD=str(library.resolve()), LD_LIBRARY_PATH=str(root),
                     KVRA_GRAPH_COUNTER_FILE=str(second)))
        assert struct.unpack('=8Q', second.read_bytes())[3:7] == (4, 1, 1, 2)
    receipt = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest(), 'command': command,
               'CPU_forwarding_verified': True, 'exact_arguments_and_error_returns_preserved': True,
               'successes_and_failures_counted_separately': True, 'GPU_used': False,
               'RTLD_LOCAL_runtime_resolution_verified': True,
               'purpose': 'untimed diagnostic actual CUDA API observation; never preload formal benchmarks'}
    (output/'CPU_BUILD_AND_FORWARDING.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    build(parser.parse_args().out)
