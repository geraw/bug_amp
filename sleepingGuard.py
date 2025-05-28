# # ====================================================
# # Problem 15: Sleeping Guard
# # Missed wake-up due to flag-only condition check
# # ====================================================
import time
import random

queue = []
waiting = False
d = []
MAX = 30
NOISE = 1.5
END = 10_000_000

def init_sleeping_guard(d_args):
    global queue, waiting, d, MAX, NOISE, END, LOOP, c1, c2
    queue = []
    global d
    queue = []
    waiting = False
    d = d_args
    MAX = 30
    NOISE = 1.5
    c1 = 2
    c2 = 1
    LOOP = 3
    END = 10_000_000

def sleeping_guard_consumer():
    global queue, waiting, d
    i = -2
    for _ in range(LOOP):
        if not queue:
            waiting = True
            yield abs(c1*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))
            while waiting:  # BUG: not rechecking actual queue state
                # print("Consumer waiting (flag = True)...")
                yield abs(c1*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))
        assert len(queue) > 0
        item = queue.pop(0)
        # print("Consumed:", item)
        yield abs(c1*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))
    yield END

def sleeping_guard_producer():
    global queue, waiting, d
    i = -1
    for _ in range(LOOP):
        queue.append("item")
        # print("Produced: item")
        yield abs(c2*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))
        if waiting:
            waiting = False  # Notifies waiting thread
        yield abs(c2*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))
    yield END

# count_pr = 0
# count = 0
# base = 0
# for _ in range(1000):
#   d = [random.randint(0, 10) for _ in range(MAX)]
#   res = simulate([sleeping_guard_consumer, sleeping_guard_producer],max_trials=1, no_found=1, init=init_sleeping_guard, init_arg= d)
#   base += res
# base/count
