END_global = 100_000  # A big number to signify the end of a thread, i.e, say that its next wake time is infinity

import numpy as np
import constants

def simulate(_threads, max_trials=10, no_found=10, init=lambda: None, init_arg=None, expected_invariant=None):
    """
    Simulates the interaction of threads over a number of trials.
    """
    
    faults = 0
    for k in range(max_trials):

        try:
            constants.count += 1
            constants.count_pr += 1
        except NameError:
            constants.count = 0
            constants.count_pr = 0       
        
        try:
            if init_arg is not None:
                init(init_arg)
            else:
                init()
                                                 # Initialize the variables
            gen = [t() for t in _threads]                       # Create generators for all threads
            wake_times = [0] * len(_threads)                    # Initial wake times for all threads
            while any(t < END_global for t in wake_times):
                nxt = np.argmin(wake_times)                    # Determine which thread to wake next
                wake_times[nxt] += next(gen[nxt])              # Run and advance the wake time for the chosen thread
                if expected_invariant is not None:
                  assert expected_invariant()                    # Check the invariant
        except AssertionError as e:
            faults += 1                                 # Increment fault counter if assertion fails
            if faults == no_found:
                return faults
    return faults