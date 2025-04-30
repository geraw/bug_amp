# Simple Counter - Dragons


MAX = 30
NOISE = 0.5
END = 10_000_000
c1 = 1
c2 = 2
LOOP = 5
in_critical_section = False
d = []

def init_dragon(d_args):
    global d, in_critical_section
    c1 = 1
    c2 = 2
    in_critical_section = False
    d = d_args
    NOISE = 0.5
    MAX = 30
    LOOP = 5

#in_critical_section = False
def critical_section_dragon(t):
  global in_critical_section

  assert not in_critical_section

  in_critical_section = True
  yield 0.5
  in_critical_section = False


def five_headed_dragon():
    global counter, d
    l = -2
    counter = 0
    yield c1*abs(d[(l := (l + 2) % MAX)] + random.uniform(-NOISE, NOISE))
    for _ in range(LOOP):
        yield c1*abs(d[(l := (l + 2) % MAX)] + random.uniform(-NOISE, NOISE))
        counter += 1
        yield c1*abs(d[(l := (l + 2) % MAX)] + random.uniform(-NOISE, NOISE))
        if counter == 5:
            yield c1*abs(d[(l := (l + 2) % MAX)] + random.uniform(-NOISE, NOISE))
            yield from critical_section_dragon(5) # Assuming critical_section is defined elsewhere
    yield END

def three_headed_dragon():
    global counter, d
    l = -1
    counter = 0
    yield c2*abs(d[(l := (l + 2) % MAX)] + random.uniform(-NOISE, NOISE))
    for _ in range(LOOP):
        yield c2*abs(d[(l := (l + 2) % MAX)] + random.uniform(-NOISE, NOISE))
        counter += 1
        yield c2*abs(d[(l := (l + 2) % MAX)] + random.uniform(-NOISE, NOISE))
        if counter == 3:
            yield c2*abs(d[(l := (l + 2) % MAX)] + random.uniform(-NOISE, NOISE))
            yield from critical_section_dragon(3) # Assuming critical_section is defined elsewhere
    yield END

base = 0
count_pr = 0
count = 0
for _ in range(1000):
    d = [random.randint(0, 5) for _ in range(MAX)]
    base += simulate([five_headed_dragon, three_headed_dragon],max_trials=1, no_found=1, init=init_dragon, init_arg= d)
print(base/count)


