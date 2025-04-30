# Semaphore

import time

# Global semaphore flag: True means available, False means acquired.
semaphore_flag = True
LOOP = 5
MAX = 20
END = 10_000_000  # A big number to signify the end of a thread, i.e, say that its next wake time is infinity
NOISE = 0.5
in_critical_section = False

def init_semaphore(d_args):
    global semaphore_flag, d
    semaphore_flag = True
    LOOP = 100
    MAX = 5
    d = d_args
    NOISE = 0.5
    in_critical_section = False
    END = 10_000_000  # A big number to signify the end of a thread, i.e, say that its next wake time is infinity

def semaphore_wait(threading):
    global semaphore_flag, d
    """
    Simulates waiting on a semaphore.
    Waits indefinitely until the semaphore (flag) is available.
    Once available, it is acquired (flag is set to False) and returns.
    """
    while not semaphore_flag:
        # time.sleep(0.01)  # Avoid busy waiting
        yield 0.1          # Avoid busy waiting
    semaphore_flag = False

def semaphore_release(t):
    """
    Releases the semaphore by setting the flag to available (True).
    """
    global semaphore_flag
    yield 0.05              # Simulate some processing time
    semaphore_flag = True


def critical_section(t):
    global in_critical_section
    # Simulated critical section.

    assert not in_critical_section

    in_critical_section = True
    yield 2              # Simulate some processing time
    in_critical_section = False


def thread0():
    global semaphore_flag, d
    """
    Simulated Thread 0:
    Repeatedly waits on the semaphore, enters the critical section, then releases the semaphore.
    """
    i = -2

    yield 4*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE)

    for _ in range (LOOP):
        yield 5*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE)
        yield from semaphore_wait(0)
        yield 5*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE)
        yield from critical_section(0)
        yield 5*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE)
        yield from semaphore_release(0)
    yield  END    # Yield control to simulate interleaving

def thread1():
    global semaphore_flag, d
    """
    Simulated Thread 1:
    Also waits indefinitely on the semaphore, enters the critical section, and then releases it.
    (In the original code this thread had a timeout, but here it is omitted.)
    """

    i = -1
    yield 4*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE)
    for _ in range (LOOP):
        yield 1*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE)
        yield from semaphore_wait(0)
        yield 1*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE)
        yield from critical_section(0)
        yield 1*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE)
        yield from semaphore_release(0)
        yield 1*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE)
    yield END

count = 0
count_pr = 0
d = [random.randint(0, 10) for _ in range(MAX)]
# Run the threads (simulated with function calls)
simulate([thread0, thread1],max_trials=100, no_found=100, init=init_semaphore, init_arg= [1,5]*MAX)
# simulate([thread0, thread1],max_trials=100, no_found=100, init=init_semaphore, init_arg= d)


