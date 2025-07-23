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
sleep_timeout = 0

def init_sleeping_guard(d_args):
    global queue, waiting, d, MAX, NOISE, END, LOOP, c1, c2, sleep_timeout
    queue = []
    global d
    queue = []
    waiting = False
    d = d_args
    MAX = 30
    NOISE = 1.5
    c1 = 4    # Consumer is very slow (higher chance to sleep too late)
    c2 = 0.4  # Producer is very fast (can sneak in and notify too early)
    LOOP = 10
    END = 10_000_000
    sleep_timeout = 500

def sleeping_guard_consumer():
    global queue, waiting, d
    i = -2
    for _ in range(LOOP):
        if not queue:
            yield abs(c1 * d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE, NOISE))
            waiting = True
            yield abs(c1 * d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE, NOISE))

            timeout = 0
            while waiting:
                yield abs(c1 * d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE, NOISE))
                timeout += 1
                if timeout > sleep_timeout:
                    assert False
                    
        assert len(queue) > 0
        yield abs(c1 * d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE, NOISE))
        item = queue.pop(0)
        yield abs(c1 * d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE, NOISE))
    yield END


def sleeping_guard_producer():
    global queue, waiting, d
    i = -1
    for _ in range(LOOP):
        yield abs(c2 * d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE, NOISE))
        queue.append("item")
        yield abs(c2 * d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE, NOISE))
        if waiting:
            yield abs(c2 * d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE, NOISE))
            waiting = False
        yield abs(c2 * d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE, NOISE))
    yield END

# count_pr = 0
# count = 0
# base = 0
# for _ in range(1000):
#   d = [random.randint(0, 10) for _ in range(MAX)]
#   res = simulate([sleeping_guard_consumer, sleeping_guard_producer],max_trials=1, no_found=1, init=init_sleeping_guard, init_arg= d)
#   base += res
# base/count
