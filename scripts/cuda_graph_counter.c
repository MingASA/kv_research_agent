/* Diagnostic-only transparent CUDA Runtime Graph launch counter. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

typedef int (*launch_fn)(void *, void *);
static uint64_t *counts;
static launch_fn runtime_launch, ptsz_launch;
static pthread_once_t once = PTHREAD_ONCE_INIT;
static _Thread_local unsigned depth;

static void initialize(void) {
    const char *path = getenv("KVRA_GRAPH_COUNTER_FILE");
    runtime_launch = (launch_fn)dlsym(RTLD_NEXT, "cudaGraphLaunch");
    ptsz_launch = (launch_fn)dlsym(RTLD_NEXT, "cudaGraphLaunch_ptsz");
    /* Python extensions can load CUDART into a local dlopen group. Resolve
       only a library already loaded by the real server; never load a backend. */
    const char *names[] = {"libcudart.so.13", "libcudart.so.12"};
    for (unsigned i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
        if (runtime_launch && ptsz_launch) break;
        void *handle = dlopen(names[i], RTLD_NOW | RTLD_NOLOAD);
        if (handle) {
            if (!runtime_launch) runtime_launch = (launch_fn)dlsym(handle, "cudaGraphLaunch");
            if (!ptsz_launch) ptsz_launch = (launch_fn)dlsym(handle, "cudaGraphLaunch_ptsz");
        }
    }
    if (!path) return;
    int fd = open(path, O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
    if (fd < 0) return;
    if (ftruncate(fd, 64) == 0) {
        void *mapping = mmap(NULL, 64, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (mapping != MAP_FAILED) {
            counts = mapping;
            counts[0] = UINT64_C(0x4b56524147524150);
            counts[1] = 1;
            counts[2] = (uint64_t)getpid();
        }
    }
    close(fd);
}

static int forward(void *graph, void *stream, int ptsz) {
    pthread_once(&once, initialize);
    launch_fn function = ptsz ? ptsz_launch : runtime_launch;
    ++depth;
    /* CUDA's shared-object-symbol-not-found error; never substitute execution. */
    int result = function ? function(graph, stream) : 302;
    if (depth == 1 && counts) {
        __atomic_fetch_add(&counts[3], 1, __ATOMIC_RELAXED);
        __atomic_fetch_add(&counts[result == 0 ? (ptsz ? 5 : 4) : 6], 1, __ATOMIC_RELAXED);
    }
    --depth;
    return result;
}

int cudaGraphLaunch(void *graph, void *stream) { return forward(graph, stream, 0); }
int cudaGraphLaunch_ptsz(void *graph, void *stream) { return forward(graph, stream, 1); }
