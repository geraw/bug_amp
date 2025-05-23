# Initialize mutexes as simple flags
import time
import random

dl_mutex1 = 0  # 0: unlocked, 1: locked
dl_mutex2 = 0  # 0: unlocked, 1: locked
timeout_deadlock = 28_000
LOOP =3
MAX = 30
END = 10_000_000  # A big number to signify the end of a thread, i.e, say that its next wake time is infinity
NOISE = 2.5
d = []
in_dl_critical_section = False
c1 = 1
c2 = 4
dl_timeout = 0.5 #sec



def init_deadlock(d_args):
    global d, k, LOOP, NOISE, END, MAX, in_dl_critical_section, c1, c2, dl_mutex1, dl_mutex2, timeout_deadlock, dl_timeout
    dl_mutex1 = 0  # 0: unlocked, 1: locked
    dl_mutex2 = 0  # 0: unlocked, 1: locked
    timeout_deadlock =30_000
    LOOP =3
    MAX = 30
    END = 10_000_000  # A big number to signify the end of a thread, i.e, say that its next wake time is infinity
    NOISE = 1.5
    d = d_args
    in_dl_critical_section = False
    c1 = 1
    c2 = 2000
    dl_timeout = 0.25 #sec
    

# Define the critical section
def dl_critical_section(thread_id):
    print(f"Thread {thread_id} entered critical section")
    yield 0.5  # Simulate some operation

# Define Thread 0's behavior
def dl_thread0():
    global dl_mutex1, dl_mutex2, timeout_deadlock, c1, c2, d, dl_timeout
    i = -1
    for _ in range(LOOP):
        # Acquire mutex1
        dl_count = 0
        dl_start_time = time.time()
        while dl_mutex1 == 1:
            yield abs(c1*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))
            assert False if dl_count > timeout_deadlock else (dl_count := dl_count + 1)
        dl_mutex1 = 1
        yield abs(c1*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))
 
        # Acquire mutex2
        dl_start_time = time.time()
        dl_count = 0
        while dl_mutex2 == 1:
            yield abs(c1*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))
            assert False if dl_count > timeout_deadlock else (dl_count := dl_count + 1)
        dl_mutex2 = 1
        yield abs(c1*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))

        # Critical section
        yield from dl_critical_section(0)
        # Release mutexes
        dl_mutex1 = 0
        yield abs(c1*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))
        dl_mutex2 = 0
        yield abs(c1*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))
    yield END

# Define Thread 1's behavior
def dl_thread1():
    global dl_mutex1, dl_mutex2, timeout_deadlock, c1, c2, d, dl_timeout
    i = -2
    for _ in range(LOOP):
          # Acquire mutex2
        dl_start_time = time.time()
        dl_count = 0
        while dl_mutex2 == 1:
            yield abs(c2*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))
            assert False if dl_count > timeout_deadlock else (dl_count := dl_count + 1)
        dl_mutex2 = 1
        yield abs(c2*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))
        # Acquire mutex1
        dl_start_time = time.time()
        dl_count = 0
        while dl_mutex1 == 1:
            yield abs(c2*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))
            assert False if dl_count > timeout_deadlock else (dl_count := dl_count + 1)
        dl_mutex1 = 1
        yield abs(c2*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))
        # Critical section
        yield from dl_critical_section(1)
        # Release mutexes
        dl_mutex2 = 0
        yield abs(c2*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))
        dl_mutex1 = 0
        yield abs(c2*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))
    yield END

# base = 0
# count_pr = 0
# count = 0
# for _ in range(1000):
#   d = [random.randint(0, 10) for _ in range(MAX)]
#   res = simulate([dl_thread0, dl_thread1 ],max_trials=1, no_found=1, init=init_deadlock, init_arg= d)
#   base += res
#   print (f'{base}/{count}') if res != 0 else None
# print(base/count)
