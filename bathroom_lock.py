# Bathroom Lock

#  A Bathroom Lock allows N genders (thread types) to
#  access a common bathroom (critical section) such that
#  different genders do not clash.

#  Bathroom Lock allows threads of the same type
#  (gender) to enter at the same time, but disallows
#  different types of threads to occupy the
#  critical section (bathroom) at the same time.

MAX = 40
NOISE = 1.5
END = 10_000_000
c1 = 1
c2 = 4
c3 = 8
c4 = 12
c5 = 1
c6 = 10
l = -5
rounds = 4

def init_bathroom(d_input, num_types=2):
    global counts, current_gender, waiting, d, l, types
    types = num_types
    counts = [0] * types  # How many of each type are in the bathroom
    waiting = [False] * types
    current_gender = None
    d = d_input
    c1 = 1
    c2 = 25
    c3 = 10
    c4 = 15
    c5 = 20
    c6 = 10
    l = -5
    rounds = 2
    NOISE = 1.5

def one_gender_allowed(type):
    return all(i == type or counts[i] == 0 for i in range(types))

# ❗ Buggy Lock Function
def lock_gender(type):
    global counts, current_gender, waiting, l

    waiting[type] = True
    yield c6*abs(d[(l := (l + 5) % MAX)] + random.uniform(-NOISE, NOISE))

    while not one_gender_allowed(type):
        yield c6*abs(d[(l := (l + 5) % MAX)] + random.uniform(-NOISE, NOISE))

    yield c6*abs(d[(l := (l + 5) % MAX)] + random.uniform(-NOISE, NOISE))

    # 🐞 Known Bug: skipping one final check after waiting can cause two genders to enter at the same time
    counts[type] += 1
    yield c6*abs(d[(l := (l + 5) % MAX)] + random.uniform(-NOISE, NOISE))
    current_gender = type
    yield c6*abs(d[(l := (l + 5) % MAX)] + random.uniform(-NOISE, NOISE))
    waiting[type] = False

def unlock_gender(type):
    global counts, current_gender, l
    yield c6*abs(d[(l := (l + 5) % MAX)] + random.uniform(-NOISE, NOISE))
    counts[type] -= 1
    yield c6*abs(d[(l := (l + 5) % MAX)] + random.uniform(-NOISE, NOISE))

    if counts[type] == 0:
        yield c6*abs(d[(l := (l + 5) % MAX)] + random.uniform(-NOISE, NOISE))
        current_gender = None
    yield c6*abs(d[(l := (l + 5) % MAX)] + random.uniform(-NOISE, NOISE))

# ✅ Critical Section with assertion
def critical_section(type):
    global l
    yield c1*abs(d[(l := (l + 5) % MAX)] + random.uniform(-NOISE, NOISE))
    # Check for other genders
    for i in range(types):
        yield c1*abs(d[(l := (l + 5) % MAX)] + random.uniform(-NOISE, NOISE))
        if i != type:
            assert counts[i] == 0, f"❌ Gender clash detected: Type {type} entered while Type {i} was present!"

# 🧑🏽‍💻 User simulation function
def gender_user_0_0():
    global d, rounds
    k = -4
    type = 0
    for _ in range(rounds):

        yield c1*abs(d[(k := (k + 5) % MAX)] + random.uniform(-NOISE, NOISE))
        yield from lock_gender(type)
        yield c1*abs(d[(k := (k + 5) % MAX)] + random.uniform(-NOISE, NOISE))
        yield from critical_section(type)
        yield c1*abs(d[(k := (k + 5) % MAX)] + random.uniform(-NOISE, NOISE))
        yield from unlock_gender(type)
        yield c1*abs(d[(k := (k + 5) % MAX)] + random.uniform(-NOISE, NOISE))

    yield END
# 🧑🏽‍💻 User simulation function
def gender_user_0_1():
    global d, rounds
    type = 0
    k = -3
    for _ in range(rounds):
        yield c2*abs(d[(k := (k + 5) % MAX)] + random.uniform(-NOISE, NOISE))
        yield from lock_gender(type)
        yield c2*abs(d[(k := (k + 5) % MAX)] + random.uniform(-NOISE, NOISE))
        yield from critical_section(type)
        yield c2*abs(d[(k := (k + 5) % MAX)] + random.uniform(-NOISE, NOISE))
        yield from unlock_gender(type)
        yield c2*abs(d[(k := (k + 5) % MAX)] + random.uniform(-NOISE, NOISE))
    yield END
# 🧑🏽‍💻 User simulation function
def gender_user_1_0():
    global d, rounds
    type = 1
    k=-2
    for _ in range(rounds):
        yield c3*abs(d[(k := (k + 5) % MAX)] + random.uniform(-NOISE, NOISE))
        yield from lock_gender(type)
        yield c3*abs(d[(k := (k + 5) % MAX)] + random.uniform(-NOISE, NOISE))
        yield from critical_section(type)
        yield c3*abs(d[(k := (k + 5) % MAX)] + random.uniform(-NOISE, NOISE))
        yield from unlock_gender(type)
        yield c3*abs(d[(k := (k + 5) % MAX)] + random.uniform(-NOISE, NOISE))
    yield END
# 🧑🏽‍💻 User simulation function
def gender_user_1_1():
    global d, rounds
    type = 1
    k = -1
    for _ in range(rounds):
        yield c4*abs(d[(k := (k + 5) % MAX)] + random.uniform(-NOISE, NOISE))
        yield from lock_gender(type)
        yield c4*abs(d[(k := (k + 5) % MAX)] + random.uniform(-NOISE, NOISE))
        yield from critical_section(type)
        yield c4*abs(d[(k := (k + 5) % MAX)] + random.uniform(-NOISE, NOISE))
        yield from unlock_gender(type)
        yield c4*abs(d[(k := (k + 5) % MAX)] + random.uniform(-NOISE, NOISE))
    yield END
# 🧑🏽‍💻 User simulation function
def gender_user_1_2():
    global d, rounds
    type = 1
    k=-1
    for _ in range(rounds):
        yield c5*abs(d[(k := (k + 5) % MAX)] + random.uniform(-NOISE, NOISE))
        yield from lock_gender(type)
        yield c5*abs(d[(k := (k + 5) % MAX)] + random.uniform(-NOISE, NOISE))
        yield from critical_section(type)
        yield c5*abs(d[(k := (k + 5) % MAX)] + random.uniform(-NOISE, NOISE))
        yield from unlock_gender(type)
        yield c5*abs(d[(k := (k + 5) % MAX)] + random.uniform(-NOISE, NOISE))
    yield END

count = 0
count_pr = 0
max_trials=1
no_found=1
base = 0
for _ in range(100):
    d = [random.randint(0, 10) for _ in range(MAX)]
    base += simulate([gender_user_0_0, gender_user_0_1, gender_user_1_0, gender_user_1_1, gender_user_1_2],max_trials=max_trials, no_found=no_found, init=init_bathroom, init_arg= d)
print(f"{base/count=}")


