# A More complex Threads

MAX = 30
NOISE = 1.5
LOOP = 5
k = -3
d1 = 2
d2 = 1
d3 = 8
END = 10_000_000  # A big number to signify the end of a thread, i.e, say that its next wake time is infinity
END_LESS = 1_000  # A big number to signify the end of a thread, i.e, say that its next wake time is infinity

def init_complex(d_args):
    global flag, mutexes, k, d , someone_in_busy_wait_complex# Added 'i' and 'd' to globals
    flag = False
    mutexes = {
        "mutex": {"by": None, "count": 0},
        "mutex2": {"by": None, "count": 0},
        "mutex3": {"by": None, "count": 0},
    }
    d = d_args
    k=-3
    someone_in_busy_wait_complex = None
    MAX = 30
    NOISE = 1.5
    LOOP = 5
    k = -3
    d1 = 2
    d2 = 1
    d3 = 8
    END = 10_000_000  # A big number to signify the end of a thread, i.e, say that its next wake time is infinity
    END_LESS = 1_000  # A big number to signify the end of a thread, i.e, say that its next wake time is infinity
    log(f'\n\n =====================\nnew session\n')


def log(msg):
    # print(msg)
    pass


def enter(t, mutex_name):
    global mutexes, someone_in_busy_wait_complex
    mutex = mutexes[mutex_name]
    log(f"Thread {t} entered enter {mutex_name}   {mutex['by']=}   {mutex['count']=}")
    if mutex["by"] is not None and mutex["by"] != t:
        loop_counter = 0
        while mutex["count"] > 0:
            if someone_in_busy_wait_complex is not None and  someone_in_busy_wait_complex != t:
                # print(f"Thread {t} entered {mutex_name}   {someone_in_busy_wait_complex} ")
                assert False
            someone_in_busy_wait_complex = t
            # Simulate waiting - in a real scenario, use a condition variable
            yield 1
            loop_counter += 1
            if loop_counter > END_LESS:
                # print(f"Thread {t} entered {mutex_name}   {someone_in_busy_wait_complex} ")
                yield END
        someone_in_busy_wait_complex = None
    mutex["count"] += 1
    mutex["by"] = t
    log(f"Thread {t} exit enter {mutex_name} {mutex['by']=}   {mutex['count']=}")

def exit(t, mutex_name):
    global mutexes
    mutex = mutexes[mutex_name]
    log(f"Thread {t} exit exit {mutex_name} {mutex['by']=}   {mutex['count']=}")
    mutex["count"] -= 1
    if mutex["count"] == 0:
        mutex["by"] = None
    log(f"Thread {t} exit exit {mutex_name} {mutex['by']=}   {mutex['count']=}")

def try_enter(t, mutex_name):
    global mutexes
    mutex = mutexes[mutex_name]
    log(f"Thread {t} try enter {mutex_name} {mutex['by']=}   {mutex['count']=}")
    if mutex["by"] is None or mutex["by"] == t:
      mutex["count"] += 1
      mutex["by"] = t
      log(f"Thread {t} try enter true {mutex_name} {mutex['by']=}   {mutex['count']=}")
      return True
    log(f"Thread {t} try enter false {mutex_name} {mutex['by']=}   {mutex['count']=}")
    return False

def thread0_complex():
    global flag, mutexes, d # Added 'i' and 'd' to globals
    i=-2
    for _ in range(LOOP):
        yield d2*abs(d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE) )
        if try_enter(0, 'mutex'):
            yield d2*abs(d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE) )
            yield from enter(0, 'mutex3')
            yield d2*abs(d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE) )
            yield from enter(0, 'mutex')
            yield d2*abs(d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE) )
            # call critical_section
            exit(0, 'mutex')
            yield d2*abs(d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE) )
            yield from enter(0, 'mutex2')
            yield d2*abs(d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE) )
            flag = False
            yield d2*abs(d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE) )
            exit(0, 'mutex2')
            yield d2*abs(d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE) )
            exit(0, 'mutex3')
            yield d2*abs(d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE) )
        else:
            yield from enter(0, 'mutex2')
            yield d2*abs(d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE) )
            flag = True
            yield d2*abs(d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE) )
            exit(0, 'mutex2')
    yield END

def thread1_complex():
    global flag, mutexes, d # Added 'i' and 'd' to globals
    i=-1
    for _ in range(LOOP):
        yield d3*abs(d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE) )
        if flag:
            yield from enter(1, 'mutex2')
            yield d3*abs(d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE) )
            yield from enter(1, 'mutex')
            yield d3*abs(d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE) )
            flag = False
            yield d3*abs(d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE) )
            # call critical_section
            yield d3*abs(d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE) )
            exit(1, 'mutex')
            yield d3*abs(d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE) )
            yield from enter(1, 'mutex2')
            yield d3*abs(d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE) )
        else:
            yield from enter(1, 'mutex')
            yield d3*abs(d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE) )
            flag = False
            yield d3*abs(d[(i := ((i + 3) % MAX))] + random.uniform(-NOISE,NOISE) )
            exit(1, 'mutex')
    yield END


base = 0
count_pr = 0
count = 0
for _ in range(1000):
  print(f"{_=}")
  d = [random.randint(0, 5) for _ in range(MAX)]
  base += simulate([thread0_complex, thread1_complex],max_trials=1, no_found=1, init=init_complex, init_arg= d)
print(base/count)


