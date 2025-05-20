# Insufficient Lock
import random
from simulate import simulate

LOOP = 2
MAX = 20
END = 10_000_000  # A big number to signify the end of a thread, i.e, say that its next wake time is infinity
NOISE = 1.5
k = 0  # Shared counter
d = []

def init_lock(d_args):
    global d, k, LOOP, NOISE
    LOOP = 3
    k = 0
    MAX = 30
    d = [ d if i%2==0 else d*100 for (i,d) in enumerate(d_args)]
    k = 0  # Shared counter
    NOISE = 1.4
    END = 10_000_000  # A big number to signify the end of a thread, i.e, say that its next wake time is infinity


def thread_lock_0():
    global k, d,LOOP, NOISE
    i = -2
    for _ in range(LOOP):
        yield abs(5*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))

        k += 2
        #critical section()
        if k == 5:
            assert False  # Simulating assertion failure

    yield END

def thread_lock_1():
    global k, d, LOOP, NOISE
    i = -1
    for _ in range(LOOP):
        yield abs(1*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))
        k -= int(1+ random.uniform(-NOISE,NOISE))
        #critical section()
    yield END


