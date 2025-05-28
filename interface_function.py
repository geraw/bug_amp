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
from general_peterson import peterson_custome1, peterson_custome2, peterson_custome3, peterson_custome4, init_peterson
from deadlock import init_deadlock, dl_thread0, dl_thread1
from waiter import init_waiter, waiter, signaler
from race import init_race_to_wait, race_to_wait_thread_1, race_to_wait_thread_2
from signalAndWait import init_signal_then_wait, signal_then_wait_thread_0, signal_then_wait_thread_1
from sleepingGuard import init_sleeping_guard, sleeping_guard_consumer, sleeping_guard_producer
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

def run_test_peterson(X, max_trials=1, no_found=1):
    constants.count_pr = 0
    constants.count    = 0
    k = simulate([peterson_custome1, peterson_custome2 ,peterson_custome3 ,peterson_custome4 ], max_trials=max_trials, no_found=no_found, init=init_peterson, init_arg=X)
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

def run_test_deadlock(X, max_trials=1, no_found=1):
    constants.count_pr = 0
    constants.count    = 0
    k = simulate([dl_thread0, dl_thread1 ],max_trials=max_trials, no_found=no_found, init=init_deadlock, init_arg= X)

    return k

def run_test_waiter(X, max_trials=1, no_found=1):
    constants.count_pr = 0
    constants.count    = 0
    k = simulate([waiter, signaler],max_trials=max_trials, no_found=no_found, init=init_waiter, init_arg= X)

    return k
def run_test_race(X, max_trials=1, no_found=1):
    global count, count_pr
    count_pr = 0
    count    = 0
    k = simulate([race_to_wait_thread_1, race_to_wait_thread_2],max_trials=max_trials, no_found=no_found, init=init_race_to_wait, init_arg= X)

    return k
def run_test_signal(X, max_trials=1, no_found=1):
    global count, count_pr
    count_pr = 0
    count    = 0
    k = simulate([signal_then_wait_thread_0, signal_then_wait_thread_1],max_trials=max_trials, no_found=no_found, init=init_signal_then_wait, init_arg= X)
    return k

def run_test_sleeping(X, max_trials=1, no_found=1):
    global count, count_pr
    count_pr = 0
    count    = 0
    k = simulate([sleeping_guard_consumer, sleeping_guard_producer],max_trials=max_trials, no_found=no_found, init=init_sleeping_guard, init_arg= X)

    return k
