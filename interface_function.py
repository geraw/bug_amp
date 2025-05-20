# Interface function


from a_more_complex_threads import init_complex, thread0_complex, thread1_complex
from boolean_flags_are_enough_for_everyone import FirstArmy, SecondArmy, initArmy
from condition_variables import Dequeue1, Dequeue2, Enqueue, init_queue
from insufficient_lock import init_lock, thread_lock_0, thread_lock_1
from simulate import simulate
from the_barrier import init_fireball, thread_1, thread_2, thread_main
from non_atomic_instructions import init_NonAtomic, thread0_NonAtomic, thread1_NonAtomic, thread2_NonAtomic
from simple_counter_dragons import five_headed_dragon, three_headed_dragon, init_dragon
from test_and_set import t11, t12, init1
import setup
import constants


S=100
def prob_our(X, max_trials=S, no_found=S):
    constants.count_pr = 0
    constants.count    = 0
    base = 0
    for _ in range(max_trials):
        base += setup.run_test(X, max_trials=1, no_found=1)
    return base/max_trials

def run_test_our(X, max_trials=1, no_found=1):
    constants.count_pr = 0
    constants.count    = 0
    k = simulate([t11, t12], max_trials=max_trials, no_found=no_found, init=init1, init_arg=X)
    return k

def run_test_boolean(X, max_trials=1, no_found=1):
    constants.count_pr = 0
    constants.count    = 0
    k = simulate([FirstArmy, SecondArmy], max_trials=max_trials, no_found=no_found, init=initArmy, init_arg=X)
    return k

def run_test_condition(X, max_trials=1, no_found=1):
    k = simulate([Enqueue, Dequeue1, Dequeue2],max_trials=max_trials, no_found=no_found, init=init_queue, init_arg= X)
    return k

def run_test_berrier(X, max_trials=1, no_found=1):
    constants.count_pr = 0
    constants.count    = 0
    k = simulate([thread_main, thread_1, thread_2], max_trials=max_trials, no_found=no_found, init=init_fireball, init_arg=X)
    return k

def run_test_semaphore(X, max_trials=1, no_found=1):
    constants.count_pr = 0
    constants.count    = 0
    k = simulate([thread_main, thread_1, thread_2], max_trials=max_trials, no_found=no_found, init=init_fireball, init_arg=X)
    return k

def run_test_lock(X, max_trials=1, no_found=1):
    constants.count_pr = 0
    constants.count    = 0
    k = simulate([thread_lock_0, thread_lock_1], max_trials=max_trials, no_found=no_found, init=init_lock, init_arg=X)
    return k

def run_test_bakery(X, max_trials=1, no_found=1):
    constants.count_pr = 0
    constants.count    = 0
    k = simulate([customer1, customer2 ,customer3 ,customer4 ], max_trials=max_trials, no_found=no_found, init=init_bakery, init_arg=X)
    return k

def run_test_non_atomic(X, max_trials=1, no_found=1):
    constants.count_pr = 0
    constants.count    = 0
    k = simulate([thread0_NonAtomic, thread1_NonAtomic, thread2_NonAtomic], max_trials=max_trials, no_found=no_found, init=init_NonAtomic, init_arg=X)
    return k

def run_test_complex(X, max_trials=1, no_found=1):
    constants.count_pr = 0
    constants.count    = 0
    k = simulate([thread0_complex, thread1_complex],max_trials=max_trials, no_found=no_found, init=init_complex, init_arg= X)
    return k

def run_test_dragons(X, max_trials=1, no_found=1):
    constants.count_pr = 0
    constants.count    = 0
    k = simulate([five_headed_dragon, three_headed_dragon],max_trials=max_trials, no_found=no_found, init=init_dragon, init_arg= X)
    return k


