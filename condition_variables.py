# Condition Variables

# def dequeue:
#   while (true) {
#     Monitor.Enter(mutex);
#     if (queue.Count == 0) {
#       Monitor.Wait(mutex);
#     }
#     queue.Dequeue();
#     Monitor.Exit(mutex);
#   }
# def enqueue:
#   while (true) {
#     Monitor.Enter(mutex);
#     queue.Enqueue(42);
#     Monitor.PulseAll(mutex);
#     Monitor.Exit(mutex);
#   }

import threading
import queue
import random

# Shared resources
mutex =True
my_queue = queue.Queue()
MAX   = 30
NOISE = 0.3
LOOP = 3

def init_queue(d_args):
    global mutex, my_queue, d
    LOOP = 2
    NOISE = 1.5
    d=d_args
    mutex = False
    my_queue = queue.Queue()
    END = 10_000_000  # A big number to signify the end of a thread, i.e, say that its next wake time is infinity


# Define the threads as classes to manage their individual states
def Dequeue1():
    global mutex, my_queue
    i = -1
    for _ in range(LOOP):

        yield 13*d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE)
        #with mutex:
        while mutex:
          yield 13*d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE)   #pass

        mutex = True
        yield 13*d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE)

        if my_queue.empty():
            yield 13*d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE)

            # mutex.wait()  # Wait if queue is empty
            mutex = False
            yield 13*d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE)
            while mutex:
                yield 13*d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE)

            mutex = True
        assert not my_queue.empty()

        yield 13*d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE)

        my_queue.get()
        yield 13*d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE)

        # mutex.notify_all()  # Notify other threads waiting on the mutex
        mutex = False
        yield 13*d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE)
    yield END

def Dequeue2():
    global mutex, queue
    i = -2
    for _ in range(LOOP):
        yield 8*d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE)
        #with mutex:
        while mutex:
          yield 8*d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE)   #pass

        mutex = True
        yield 8*d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE)

        if my_queue.empty():
            yield 8*d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE)

            # mutex.wait()  # Wait if queue is empty
            mutex = False
            yield 8*d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE)
            while mutex:
                yield 8*d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE)

            mutex = True

        assert not my_queue.empty()
        yield 8*d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE)

        my_queue.get()
        yield 8*d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE)

        # mutex.notify_all()  # Notify other threads waiting on the mutex
        mutex = False
        yield 8*d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE)
    yield END


def Enqueue():
    global mutex, my_queue
    i = -3
    for _ in range(2*LOOP):
        yield d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE)

        #with mutex:
        while mutex:
            yield d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE)

        mutex = True
        yield d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE)

        my_queue.put(42)

        yield d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE)
        # mutex.notify_all()  # Notify other threads waiting on the mutex
        mutex = False
        yield d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE)
    yield END


