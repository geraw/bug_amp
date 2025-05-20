# Test and Set

#nonAtomicTestAndSet 1 - test - x==3
import random

X0 = 1                      # Initial value of x
TARGET = 3                  # Value to compare with x
END = 10_000_000  # A big number to signify the end of a thread, i.e, say that its next wake time is infinity
LOOP = 1
MAX = 20
NOISE = 0.7

def init1(arg_d):
    global x, d
    x = X0            # TODO- to take from the argument list
    d = arg_d
    TARGET = 3                  # Value to compare with x
    LOOP = 1
    MAX = 20
    NOISE = 0.5

def t11():
  global x, d
  i = -2
  for i in range(LOOP):

      # yield d[0]+0.5   # Simulate waiting time
      yield 0.95*abs(d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE) )

      x=TARGET
#      yield d[1]+0.5   # Simulate waiting time
      yield 0.95*abs(d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))

      if x != TARGET:
          # yield d[2]+0.5   # Simulate waiting time
          yield 0.95*abs(d[(i := ((i + 2) % MAX))] + random.uniform(-NOISE,NOISE))
          assert (x!=TARGET)
  yield END

# 4 < 2 + D < 5
# D = 2.5

def t12():
  global x, d
  for (i,di) in enumerate(d):
      # yield  1.75+random.uniform(-0.15, 0.5) # Simulate waiting time
      yield 1.3 + 3.5 * di + random.uniform(-NOISE,NOISE)

      x = i
  yield END


# base = 0
# count_pr = 0
# count = 0
# for _ in range(100):
#   d = [random.randint(0, 10) for _ in range(MAX)]
#   base += simulate([t11, t12],max_trials=100, no_found=100, init=init1, init_arg= d)
# print(base/count)


