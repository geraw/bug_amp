import random
import time



def init_atomicity_bypass(d_args):
    global d, c1, c2, counter, mutex, END, MAX, NOISE, LOOP,  in_critical_section
    d = d_args
    c1 = 1.0
    c2 = 5.0
    NOISE = 2.5
    counter = 0
    mutex = 0
    END = 10_000_000
    in_critical_section = False
    MAX = 30
    LOOP = 2

def logg(x):
  # print(x)
  pass


def atomicity_thread_a():
    global counter, mutex, d, c1
    i = -2
    for _ in range(LOOP):
        # Wait for lock
        while mutex:
            yield abs(c1 * d[(i := (i + 2) % MAX)] + random.uniform(-NOISE, NOISE))

        mutex = 1  # Acquire lock
        yield abs(c1 * d[(i := (i + 2) % MAX)] + random.uniform(-NOISE, NOISE))

        local = counter
        yield abs(c1 * d[(i := (i + 2) % MAX)] + random.uniform(-NOISE, NOISE))

        mutex = 0  # Bug: premature unlock!
        yield abs(0.5 * c1 * d[(i := (i + 2) % MAX)] + random.uniform(-NOISE, NOISE))
        assert counter == local

        counter = local + 1  # Unsafe write

        # call the atomicity_critical_section("A")

    yield END

def atomicity_thread_b():
    global counter, mutex, d, c2
    i = -1
    for _ in range(LOOP):
        while mutex:
            yield abs(c2 * d[(i := (i + 2) % MAX)] + random.uniform(-NOISE, NOISE))

        mutex = 1
        yield abs(c2 * d[(i := (i + 2) % MAX)] + random.uniform(-NOISE, NOISE))

        local = counter
        yield abs(c2 * d[(i := (i + 2) % MAX)] + random.uniform(-NOISE, NOISE))

        mutex = 0  # Bug here as well: releasing early
        yield abs(0.01 * c2 * d[(i := (i + 2) % MAX)] + random.uniform(-NOISE, NOISE))
        assert counter == local
        counter = local + 1

        # call the atomicity_critical_section("A")

    yield END
    
# count_pr = 0
# count = 0
# base = 0
# for _ in range(1000):
#     d = [random.randint(0, 10) for _ in range(MAX)]
#     res = simulate([atomicity_thread_a, atomicity_thread_b],max_trials=1, no_found=1, init=init_atomicity_bypass, init_arg= d)
#     base += res
#     # print(f'{base}/{count}') if res!=0 else None
# print (f'{base/count}')
