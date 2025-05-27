import random

# Shared state
flag = False
waiting_threads = set()
d = []
c1 = 0
c2 = 0
c3 =0
i = 0
MAX = 30
END = 10_000_000
NOISE = 0
LOOP = 0

def init_waiter(d_args):
    global flag, waiting_threads, d, c1, c2, c3, i, MAX, END, NOISE, LOOP
    flag = False
    waiting_threads = set()
    d = d_args
    c1 = 1
    c2 = 50
    c3 = 0.1
    i = 0
    MAX = 30
    LOOP = 10
    NOISE = 1.5
    END = 10_000_000

def logg(x):
  # print(x)
  pass

# Simulated cv.wait() using yield — buggy: waits inside cv_wait only once caller calls it
def cv_wait(thread_id):
    global flag, d, i
    
    logg(f"[cv_wait] Thread {thread_id} waiting (simulated)")
    yield c3*abs(d[(i := ((i + 1) % MAX))] + random.uniform(-NOISE,NOISE) )

    waiting_threads.add(thread_id)
    yield c3*abs(d[(i := ((i + 1) % MAX))] + random.uniform(-NOISE,NOISE) )

    while not flag:          # Loop inside wait function itself (BUG)
        yield c3*abs(d[(i := ((i + 1) % MAX))] + random.uniform(-NOISE,NOISE) )

    waiting_threads.discard(thread_id)
    logg(f"[cv_wait] Thread {thread_id} resumed")

# Buggy waiter: calls cv_wait() once, no loop guarding the condition
def waiter():
    global flag, d, i, c1
    thread_id = 0
    logg(f"[Waiter] Thread {thread_id} started")

    for _ in range(LOOP):
        yield c1*abs(d[(i := ((i + 1) % MAX))] + random.uniform(-NOISE,NOISE) )

        if not flag:
            yield c1*abs(d[(i := ((i + 1) % MAX))] + random.uniform(-NOISE,NOISE) )
            assert flag == False, "[BUG] flag was already True before wait — signal lost?"

            yield from cv_wait(thread_id)  # only waits once (calls wait once)

        yield c1*abs(d[(i := ((i + 1) % MAX))] + random.uniform(-NOISE,NOISE) )

        logg(f"[Waiter] Thread {thread_id} proceeds assuming flag == True")
        logg(f"[Waiter] Thread {thread_id} done")
    yield END

# Signaler sets flag and resumes all waiting threads
def signaler():
    global flag, d, i, c2
    thread_id = 1
    logg(f"[Signaler] Thread {thread_id} started")
    for _ in range(LOOP):
        yield c2*abs(d[(i := ((i + 1) % MAX))] + random.uniform(-NOISE,NOISE) )

        # Simulate random delay
        for _ in range(random.randint(0, 5)):
            yield c2*abs(d[(i := ((i + 1) % MAX))] + random.uniform(-NOISE,NOISE) )
            logg(f"[Signaler] Thread {thread_id} working...")
            

        flag = True
        yield c2*abs(d[(i := ((i + 1) % MAX))] + random.uniform(-NOISE,NOISE) )
        logg(f"[Signaler] Set flag = True")
        logg(f"[Signaler] Notifying all waiting threads")

        # Simulate notify_all
        waiting_threads.clear()

        logg(f"[Signaler] Thread {thread_id} done")
    yield END

# base = 0
# count_pr = 0
# count = 0
# for _ in range(1000):
#     d = [random.randint(0, 10) for _ in range(MAX)]
#     base += simulate([waiter, signaler],max_trials=1, no_found=1, init=init_waiter, init_arg= d)
# print(base/count)
