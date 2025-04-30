# Insufficient Lock

import random
from simulate import simulate

# Global flag: True means available, False means acquired.
LOOP = 5
MAX = 20
END = 10_000_000  # A big number to signify the end of a thread, i.e, say that its next wake time is infinity
NOISE = 0.5
flag = False  # Used as a manual lock
k = 0  # Shared counter
in_critical_section = False
d = []

def init_lock(d_args):
    global flag, d, k, in_critical_section
    flag = False  # Used as a manual lock
    LOOP = 2
    k = 0
    MAX = 30
    d = [ d if i%2==0 else d*100 for (i,d) in enumerate(d_args)]
    k = 0  # Shared counter
    NOISE = 0.4
    in_critical_section = False
    END = 10_000_000  # A big number to signify the end of a thread, i.e, say that its next wake time is infinity


def critical_section_lock(t):
    global in_critical_section
    # Simulated critical section.

    assert not in_critical_section

    in_critical_section = True
    yield 0.3              # Simulate some processing time
    in_critical_section = False


def thread_lock_0():
    global flag, k, d
    i = -2
    for _ in range(LOOP):
        yield abs(5*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))

        while flag:  # Wait until flag is False (lock is free)
            yield abs(5*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))

        flag = True  # Acquire lock
        yield abs(5*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))

        k += 2

        yield abs(5*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))

        yield from critical_section_lock(0)

        yield abs(5*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))

        if k == 5:
            assert False  # Simulating assertion failure

        abs(5*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))

        flag = False  # Release lock
    yield END

def thread_lock_1():
    global flag, k, d
    i = -1
    for _ in range(LOOP):
        yield abs(1*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))
        while flag:  # Wait until flag is False (lock is free)
            yield abs(1*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))

        flag = True  # Acquire lock
        yield abs(1*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))
        k -= 1
        yield abs(1*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))

        yield from critical_section_lock(0)
        yield abs(1*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))

        flag = False  # Release lock
    yield END

simulate([thread_lock_0, thread_lock_1],max_trials=100, no_found=100, init=init_lock, init_arg= [1, .01]*MAX)



import time

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
base = 0

count_pr = 0
count = 0
for _ in range(100):
  d = [random.randint(0, 10) for _ in range(MAX)]
  base += simulate([thread_lock_0, thread_lock_1],max_trials=100, no_found=100, init=init_lock, init_arg= d)
print(base/count)

