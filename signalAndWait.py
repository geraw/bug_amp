import random

flag = False
mutex = 0  # simulate lock
wait_blocked = False
timeout = 1000
MAX = 30
NOISE = 1.5
d = []


def init_signal_then_wait(d_args):
    global flag, mutex, wait_blocked, timeout, MAX, NOISE, d, LOOP, END, c1, c2
    LOOP = 1
    flag = False
    mutex = 0
    wait_blocked = False
    timeout = 100
    MAX = 30
    NOISE = 2.5
    d = d_args
    c1 = 1
    c2 = 4
    END = 10_000_000  # A big number to signify the end of a thread, i.e, say that its next wake time is infinity

def logg(x):
  # print (x)
  pass  

def signal_then_wait_thread_0():
    global flag, wait_blocked, mutex, d
    i = -2
   
    for _ in range(LOOP):
        yield abs(c1 * d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE, NOISE))  # simulate small delay
        logg(f'Tread 0 start')
        while mutex: 
            yield abs(c1 * d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE, NOISE))
        # acquire lock
        mutex = 1
        yield abs(5*c1 * d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE, NOISE))  # simulate small delay
        logg("Thread 0 acquired lock ")
        attempts = 0
        while not flag:
            wait_blocked = True
            logg("Thread 0 waiting on condition")
            yield abs(c1 * d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE, NOISE))
            attempts += 1
            if attempts > timeout:
                assert False, "Thread 0 deadlocked waiting for signal"
        wait_blocked = False
        logg("Thread 0 proceeds")
        mutex = 0
        yield abs(c1 * d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE, NOISE))
    yield END

def signal_then_wait_thread_1():
    global flag, mutex, d
    i = -1

    for _ in range(LOOP):
        yield abs(10*c2 * d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE, NOISE))
        logg(f'Tread 1 start')

        flag = True
        logg("Thread 1 set flag = true")
        yield abs(c2 * d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE, NOISE))
        # yield 0.01
        while mutex: 
            yield 0.01

        # yield abs(c2 * d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE, NOISE))
        yield 0.01

        mutex = 1
        yield abs(c2 * d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE, NOISE))
        logg("Thread 1 acquired lock")
        logg("Thread 1 notifies")
        mutex = 0
        logg("Thread 1 releaseed lock")

        yield abs(c2 * d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE, NOISE))
    yield END
# Run it

# count_pr = 0
# count = 0
# base = 0
# for _ in range(1000):
#   d = [random.randint(0, 10) for _ in range(MAX)]
#   res = simulate([signal_then_wait_thread_0, signal_then_wait_thread_1],max_trials=1, no_found=1, init=init_signal_then_wait, init_arg= d)
#   base += res
# base/count
