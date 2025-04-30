# Non-Atomic Instructions

# prompt: Thread 0
# a = a + 1;
#   temp = a + 1;
#   a = temp;
# if (a == 1) {
#   critical_section();
# }
# Thread 1
# // Expand the following instruction:
# a = a + 1;
# if (a == 1) {
#   critical_section();
# }
# translate to python

LOOP = 5
in_critical_section = False
a = 0
d = []
MAX = 30
NOISE = 0.5
d1 = 1
d2 = 3
d3 = 7
END = 10_000_000  # A big number to signify the end of a thread, i.e, say that its next wake time is infinity


def init_NonAtomic(d_args):
    global a, d, LOOP,  in_critical_section
    a = 0
    d = d_args
    LOOP = 4
    END = 10_000_000  # A big number to signify the end of a thread, i.e, say that its next wake time is infinity
    MAX = 30
    NOISE = 0.5
    in_critical_section = False

#in_critical_section = False
def critical_section_NonAtomic(t):
  global in_critical_section

  assert not in_critical_section

  in_critical_section = True
  # print(f"Thread {t} entered critical section")
  yield 1
  in_critical_section = False


def thread0_NonAtomic():
    global a, d, LOOP,  in_critical_section
    i = -3

    for _ in range(LOOP):
        yield d1*abs(d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE) )
        # a = a + 1
        temp = a + 1
        yield d1*abs(d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE) )
        a = temp
        yield d1*abs(d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE) )
        if a == 1:
            yield d1*abs(d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE) )
            yield from critical_section_NonAtomic(0) # Replace with your actual critical section code
    yield END

def thread1_NonAtomic():
    global a, d, LOOP,  in_critical_section
    i = -2

    for _ in range(LOOP):
        yield d2*abs(d[(i := (((i + 3) % MAX)))] + random.uniform(-NOISE,NOISE) )

        # a = a + 1
        temp = a + 1
        yield d2*abs(d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE) )
        a = temp
        yield d2*abs(d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE) )
        if a == 1:
            yield d2*abs(d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE) )
            yield from critical_section_NonAtomic(1) # Replace with your actual critical section code
    yield END

def thread2_NonAtomic():
    global a, d, LOOP,  in_critical_section
    i = -1

    for _ in range(LOOP):
        yield d3*abs(d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE) )

        # a = a + 1
        temp = a + 1
        yield d3*abs(d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE) )
        a = temp
        yield d3*abs(d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE) )
        if a == 1:
            yield d3*abs(d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE) )
            yield from critical_section_NonAtomic(2) # Replace with your actual critical section code
    yield END

base = 0
count_pr = 0
count = 0
for _ in range(100):
  d = [random.randint(0, 10) for _ in range(MAX)]
  base += simulate([thread0_NonAtomic, thread1_NonAtomic, thread2_NonAtomic],max_trials=100, no_found=100, init=init_NonAtomic, init_arg= d)
print(base/count)


