# # ====================================================
# # Problem 16: Phantom Permit
# # Semaphore misused: release without successful wait
# # ====================================================

import random

semaphore = 1  # Acts like a binary semaphore (1 = available)
MAX = 0
NOISE = 0
END = 10_000_000
LOOP = 0
d = []

def logg(x):
  # print (x)
  pass  

def init_phantom_permit(d_args):
    global semaphore, MAX, NOISE, END, d, c1, c2, PERMIT_TIMEOUT, LOOP, in_critical_section
    semaphore = 1
    MAX = 30
    NOISE = 2
    END = 10_000_000
    d = d_args
    c1 = 30
    c2 = 3 # 6
    LOOP = 2
    in_critical_section = False



def critical_section_permit(t):
    global in_critical_section
    # Simulated critical section.

    assert not in_critical_section

    in_critical_section = True
    yield 1              # Simulate some processing time
    in_critical_section = False




def phantom_permit_thread_0():
    global semaphore, d, c1, PERMIT_TIMEOUT
    i = -2
    for _ in range(LOOP):
        # yield abs(c1*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))
        while semaphore <= 0:
            yield abs(c1 * d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE, NOISE))
        semaphore -= 1                    # Acquire permit
        yield abs(c1*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))
        logg("Thread 0 entered critical section")
        yield from critical_section_permit(0.1)
        yield abs(c1*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))
        semaphore += 1                    # Release permit
        yield abs(c1*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))
    yield END

def phantom_permit_thread_1():
    global semaphore, d, c2
    i = -1
    for _ in range(LOOP):
        yield abs(c2 * d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE, NOISE))

        while semaphore <= 0:
            yield abs(c2 * d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE, NOISE))

        logg("Thread 1 entered critical section")
        yield from critical_section_permit(0.1)
        yield abs(c1*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))
        semaphore += 1  # BUG: releasing permit without acquiring
        logg("Thread 1 released (phantom permit)")
        yield abs(c2 * d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE, NOISE))

    yield END

# count_pr = 0
# count = 0
# base = 0
# for _ in range(1000):
#     d = [random.randint(0, 5) for _ in range(MAX)]
#     res = simulate([phantom_permit_thread_0, phantom_permit_thread_1],max_trials=1, no_found=1, init=init_phantom_permit, init_arg= d)
#     base += res
#     # print(f'{base}/{count}') if res!=0 else None
# print (f'{base/count}')
