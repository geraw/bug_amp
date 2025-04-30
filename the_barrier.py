# The Barrier

import random

# Shared variables
fireballCharge = 0
barrier_count = 0  # Acts as a synchronization barrier
WAIT_LIMIT = 2      # Number of "threads" to wait before proceeding
MAX = 30
LOOP = 5
NOISE = 3

def init_fireball(d_args):
    global fireballCharge, barrier_count, d
    fireballCharge = 0
    barrier_count = 0  # Acts as a synchronization barrier
    WAIT_LIMIT = 2      # Number of "threads" to wait before proceeding
    MAX = 30
    LOOP = 5
    d = d_args
    NOISE = 3
    END = 10_000_000  # A big number to signify the end of a thread, i.e, say that its next wake time is infinity


def barrier_signal_and_wait(t):
    """ Simulates a barrier where all functions must reach before continuing. """
    global barrier_count, d
    barrier_count += 1

    # Wait until all processes reach the barrier
    while barrier_count < WAIT_LIMIT:
      yield 0.5

def fireball(t):
    """ Simulated fireball function """
    yield 0.05

def thread_main():
    """ Simulates the main thread logic """
    global fireballCharge, barrier_count, d
    i=-3

    yield d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE)

    for _ in range(LOOP):

        yield 1*d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE)

        fireballCharge += 1

        yield 1*d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE)

        yield from barrier_signal_and_wait(0)

        yield 1*d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE)

        if fireballCharge < 2:
            assert False

        yield 1*d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE)

        yield from fireball(0)
    yield END

def thread_1():
    """ Simulates the first worker thread """
    global fireballCharge, barrier_count, d
    i = -2

    for _ in range(LOOP):

        yield d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE)

        fireballCharge += 1

        yield d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE)

        yield from barrier_signal_and_wait(0)

        yield d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE)

    yield END


def thread_2():
    """ Simulates the second worker thread """
    global fireballCharge, barrier_count
    i = -1

    for _ in range(LOOP):

        yield 15*d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE)

        fireballCharge += 1

        yield 15*d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE)

        yield from barrier_signal_and_wait(0)

        yield 15*d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE)

        yield from barrier_signal_and_wait(0)  # Extra wait

        yield 15*d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE)

        fireballCharge = 0  # Reset after firing

        yield 15*d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE)

    yield END

count = 0
count_pr = 0
vector = [random.randint(0, 10) for _ in range(MAX)]
# Run the threads (simulated with function calls)
# simulate([thread_main, thread_1, thread_2],max_trials=100, no_found=100, init=init_fireball, init_arg= [1,3,4]*MAX)
# simulate([thread_main, thread_1, thread_2],max_trials=100, no_found=100, init=init_fireball, init_arg= vector)


