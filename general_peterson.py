import random

NUM_CUSTOMERS = 4
levels = [0] * NUM_CUSTOMERS
last_to_enter = [None] * (NUM_CUSTOMERS - 1)
LOOP =0
MAX = 30
END = 10_000_000  # A big number to signify the end of a thread, i.e, say that its next wake time is infinity
NOISE = 0
d = []
in_peterson_critical_section = False
c1 = 0
c2 = 0
c3 = 0
c4 = 0
cc1 = 0
cc2 = 0


def init_peterson(d_args):
    global d, k, LOOP, NOISE, END, MAX, NUM_CUSTOMERS, in_peterson_critical_section, c1, c2, c3, c4, cc1, cc2, levels, last_to_enter
    NUM_CUSTOMERS = 4
    levels = [0] * NUM_CUSTOMERS
    last_to_enter = [None] * (NUM_CUSTOMERS - 1)
    LOOP =10
    MAX = 30
    END = 10_000_000  # A big number to signify the end of a thread, i.e, say that its next wake time is infinity
    NOISE = 1.5
    d = d_args
    in_peterson_critical_section = False
    c1 = 1
    c2 = 15
    c3 = 6
    c4 = 4
    cc1 = 1
    cc2 = 1
    
#in_critical_section = False
def peterson_critical_section(t):
  global in_peterson_critical_section
  assert not in_peterson_critical_section
  in_peterson_critical_section = True
  yield 0.5
  in_peterson_critical_section = False


def peterson_custome1():
    i = -1
    for _ in range(LOOP):
        yield abs(c1*d[(i := ((i + 4) % MAX))] + random.uniform(-NOISE,NOISE))
        for level in range(NUM_CUSTOMERS - 1):
            last_to_enter[level] = 0  # Bug: incorrect order
            yield abs(cc1*c1*d[(i := ((i + 4) % MAX))] + random.uniform(-NOISE,NOISE))
            levels[0] = level
            yield abs(c1*d[(i := ((i + 4) % MAX))] + random.uniform(-NOISE,NOISE))

            while any(
                (other != 0 and levels[other] >= level and last_to_enter[level] == 0)
                for other in range(NUM_CUSTOMERS)
            ):
                  yield abs(c1*d[(i := ((i + 4) % MAX))] + random.uniform(-NOISE,NOISE))
        # Critical section
        yield abs(c1*d[(i := ((i + 4) % MAX))] + random.uniform(-NOISE,NOISE))
        yield from peterson_critical_section(0)
        levels[0] = -1
        yield abs(c1*d[(i := ((i + 4) % MAX))] + random.uniform(-NOISE,NOISE))
    yield END


def peterson_custome2():
    i = -2
    for _ in range(LOOP):
        yield abs(c2*d[(i := ((i + 4) % MAX))] + random.uniform(-NOISE,NOISE))
        for level in range(NUM_CUSTOMERS - 1):
            last_to_enter[level] = 1  # Bug: incorrect order
            yield abs(cc2*c2*d[(i := ((i + 4) % MAX))] + random.uniform(-NOISE,NOISE))
            levels[1] = level
            yield abs(c2*d[(i := ((i + 4) % MAX))] + random.uniform(-NOISE,NOISE))

            while any(
                (other != 1 and levels[other] >= level and last_to_enter[level] == 1)
                for other in range(NUM_CUSTOMERS)
            ):
                  yield abs(c2*d[(i := ((i + 4) % MAX))] + random.uniform(-NOISE,NOISE))
        # Critical section
        yield abs(c2*d[(i := ((i + 4) % MAX))] + random.uniform(-NOISE,NOISE))
        yield from peterson_critical_section(1)
        levels[1] = -1
        yield abs(c2*d[(i := ((i + 4) % MAX))] + random.uniform(-NOISE,NOISE))
    yield END

def peterson_custome3():
    i = -3
    for _ in range(LOOP):
        yield abs(c3*d[(i := ((i + 4) % MAX))] + random.uniform(-NOISE,NOISE))
        for level in range(NUM_CUSTOMERS - 1):
            last_to_enter[level] = 2  # Bug: incorrect order
            yield abs(cc1*c3*d[(i := ((i + 4) % MAX))] + random.uniform(-NOISE,NOISE))
            levels[2] = level
            yield abs(c3*d[(i := ((i + 4) % MAX))] + random.uniform(-NOISE,NOISE))

            while any(
                (other != 2 and levels[other] >= level and last_to_enter[level] == 2)
                for other in range(NUM_CUSTOMERS)
            ):
                  yield abs(c3*d[(i := ((i + 4) % MAX))] + random.uniform(-NOISE,NOISE))
        # Critical section
        yield abs(c3*d[(i := ((i + 4) % MAX))] + random.uniform(-NOISE,NOISE))
        yield from peterson_critical_section(2)
        levels[2] = -1
        yield abs(c3*d[(i := ((i + 4) % MAX))] + random.uniform(-NOISE,NOISE))
    yield END

def peterson_custome4():
    i = -4
    for _ in range(LOOP):
        yield abs(c4*d[(i := ((i + 4) % MAX))] + random.uniform(-NOISE,NOISE))
        for level in range(NUM_CUSTOMERS - 1):
            last_to_enter[level] = 3  # Bug: incorrect order
            yield abs(c4*d[(i := ((i + 4) % MAX))] + random.uniform(-NOISE,NOISE))
            levels[3] = level
            yield abs(c4*d[(i := ((i + 4) % MAX))] + random.uniform(-NOISE,NOISE))

            while any(
                (other != 3 and levels[other] >= level and last_to_enter[level] == 3)
                for other in range(NUM_CUSTOMERS)
            ):
                  yield abs(c4*d[(i := ((i + 4) % MAX))] + random.uniform(-NOISE,NOISE))
        # Critical section
        yield abs(c4*d[(i := ((i + 4) % MAX))] + random.uniform(-NOISE,NOISE))
        yield from peterson_critical_section(4)
        levels[3] = -1
        yield abs(c4*d[(i := ((i + 4) % MAX))] + random.uniform(-NOISE,NOISE))
    yield END


# base = 0
# count_pr = 0
# count = 0
# for _ in range(1000):
#   d = [random.randint(0, 10) for _ in range(MAX)]
#   base += simulate([peterson_custome1, peterson_custome2 ,peterson_custome3 ,peterson_custome4 ],max_trials=1, no_found=1, init=init_peterson, init_arg= d)
#   # print (f'{base}/{count}')
# print(base/count)
