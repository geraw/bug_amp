import time
import random

# ====================================================
# Problem 13: Race-to-Wait
# Deadlock due to non-atomic coordination on waiters
# ====================================================
waiters = 0
MAX = 30
c1 = 1
c2 = 0
NOISE = 0
LOOP = 1
END = 10_000_000  # A big number to signify the end of a thread, i.e, say that its next wake time is infinity
MAX_WAIT = 0

def init_race_to_wait(d_args):
    global waiters, d, MAX, c1, c2, LOOP, END, NOISE, MAX_WAIT
    waiters = 0
    d = d_args
    MAX = 30
    c1 = 1
    c2 = 250
    NOISE = 2
    LOOP = 1
    MAX_WAIT = 1500
    END = 10_000_000  # A big number to signify the end of a thread, i.e, say that its next wake time is infinity

def logg(x):
  # print (x)
  pass

def race_to_wait_thread_1():
    global waiters, d, MAX, c1
    i = -2
    for _ in range(LOOP):
        yield abs(c1*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))     # Simulate preemption before reading
        temp = waiters              # BUG: read not atomic
        yield abs(c1*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))     # Context switch could happen here
        waiters = temp + 1          # BUG: write based on stale value
        logg("Thread 1 incremented to:")
        # Deadlock: both threads wait for waiters == 2, but it never happens
        attempts = 0
        yield abs(c1*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))     # Context switch could happen here
        while waiters < 2:
            logg("Thread 1 waiting...")
            attempts += 1
            yield abs(c1*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))     # Context switch could happen here
            if attempts > MAX_WAIT:
                logg("Thread 1: Deadlock detected!")
                assert False
            yield abs(c1*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))     # Context switch could happen here

    yield END

def race_to_wait_thread_2():
    global waiters, d, MAX, c2
    i = -1
    for _ in range(LOOP):
        yield abs(c2*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))     # Simulate preemption before reading
        temp = waiters              # Same non-atomic read
        yield abs(c2*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))     # Simulate preemption before reading
        waiters = temp + 1          # Overwrites previous write
        logg("Thread 2 incremented to:")
        attempts = 0
        yield abs(c2*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))     # Simulate preemption before reading
        while waiters < 2:
            logg("Thread 1 waiting...")
            attempts += 1
            yield abs(c2*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))     # Simulate preemption before reading
            if attempts > MAX_WAIT:
                logg("Thread 1: Deadlock detected!")
                assert False
            yield abs(c2*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))     # Simulate preemption before reading
    yield END


# count_pr = 0
# count = 0
# base = 0
# for _ in range(100):
#   d = [random.randint(0, 10) for _ in range(MAX)]
#   res = simulate([race_to_wait_thread_1, race_to_wait_thread_2],max_trials=1, no_found=1, init=init_race_to_wait, init_arg= d)
#   base += res
#   # print(f'{base}/{count}') if res!=0 else None
# print(base/count)