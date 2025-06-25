# Boolean Flags Are Enough For Everyone

import random

LOOP     = 1
MAX   = 20
NOISE = 1.5

def initArmy(arg_d):
  global flag, d, i, in_critical_section, END, LOOP, MAX, NOISE
  d    = arg_d
  flag = False
  in_critical_section = False
  END = 10_000_000  # A big number to signify the end of a thread, i.e, say that its next wake time is infinity
  LOOP     = 1
  MAX   = 20
  NOISE = 1.5



#in_critical_section = False
def critical_section(t):
  global in_critical_section

  assert not in_critical_section

  in_critical_section = True
  yield 0.5
  in_critical_section = False



def FirstArmy():
  global flag, d
  i = -2
  for _ in range(LOOP):

      yield 6*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE)

      while flag:
          yield 6*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE)    # Busy wait

      yield 6*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE)

      flag = True

      yield 6*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE)

      yield from critical_section(0)

      yield 6*d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE)

      flag = False

  yield END

def SecondArmy():
  global flag, d
  i = -1
  for _ in range(LOOP):
      yield d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE)

      while flag:
          yield d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE)   # Busy wait

      yield d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE)

      flag = True

      yield d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE)

      yield from critical_section(1)

      yield d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE)

      flag = False

  yield END

count = 0
count_pr = 0

#simulate([FirstArmy, SecondArmy],max_trials=100, no_found=100, init=init, init_arg= [1,1,10,10] + [1]*20)
# simulate([FirstArmy, SecondArmy],max_trials=100, no_found=100, init=initArmy, init_arg= [1,3.6]*20)


