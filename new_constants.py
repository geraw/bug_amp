import numpy as np
import sys
import random

from interface_function import * # Assuming this provides run_test_* and prob_our
# You'll need to ensure all run_test_* functions for the problems below are defined in interface_function.py
# Example:
# def run_test_atomicity_bypass(x, **kwargs):
#     # ... implementation ...
# def prob_atomicity_bypass(x, **kwargs):
#     # ... implementation ...
# Or if all use prob_our and run_test_our, then that's fine.

M_CORRELETION_THRESHOLD = 0.10    # Magalit correlation thrashhold
S_CORRELETION_THRESHOLD = 0.20    # Spearman correlation threshhold
n_informative = 7
probability_factor = 1
n_features = 50 # This will be overwritten by constants.probs
cost = 0
N_TEST = 1_000                     # Number of random elements for test the coore;etion
count = 0
count_pr = 0
cost = 0
random_state = 42
multip = 1 # This will be overwritten by constants.probs
B=30
S=40
TOP_IN_MARGALITS_CORRELATION = 100
# bounds = [(0, multip) for _ in range(n_features)] # This should be dynamically set based on each problem
MAX_TRIALS = 30
N_INITIAL_SAMPLES = 6
END = 10_000_000

rng = np.random.RandomState(random_state)
file_path = f'reports/'

NUM_TO_CHECK = 20
NUM_OF_TESTS = 50          # number of tests for calculate the AVR and SDT of the methods
N_TRAIN = 100                   # Number of random elements (increments) that the classifier is trained
N_PARALLEL = 1 # Keep this as 1 for now, or adapt as per your system's capability

# Access individual arguments (This part will be handled by argparse in run_classifier.py)
# if len(sys.argv) > 1:
#     res_ver = sys.argv[1]
# else:
#     res_ver = random.randint(0, 1000)

csv_file_path = f'reports/'
# csv_file_name and csv_ab_file_name will be dynamically set in run_classifier.py
# csv_file_name = f'results_{res_ver}.csv'
# csv_ab_file_name = f'results_ab_{res_ver}.csv'


file_data = f'{M_CORRELETION_THRESHOLD}_{S_CORRELETION_THRESHOLD}_{random_state}_clf_model.pkl'  # Replace with your desired path

# Full list of problem names based on the document and your commented out lines
# Ensure that corresponding run_test_* and prob_* functions are available for each.
ALL_PROBLEM_NAMES_LIST = [
    'atomicity_bypass',
    'broken_barrier', # Renamed from 'Barrier' for consistency
    'broken_peterson', # Renamed from 'peterson' for consistency
    'delayed_write', # Added based on document
    'flagged_deadlock', # Added based on document
    'if_not_while', # Renamed from 'Non_atomic' for consistency, assuming this is the one
    'lock_order_inversion', # Added based on document
    'lost_signal', # Renamed from 'signal' for consistency
    'partial_lock', # Added based on document
    'phantom_permit',
    'race_to_wait', # Renamed from 'waiter' and 'race' for consistency, assuming these are covered
    'racy_increment', # Added based on document
    'semaphore_leak', # Renamed from 'semaphore' for consistency
    'shared_counter', # Added based on document
    'shared_flag', # Added based on document
    'signal_then_wait', # Assuming this is 'signal' if renamed
    'sleeping_guard', # Renamed from 'sleeping' for consistency
]

# A dictionary mapping simplified names to the actual 'probs' list entries.
# You MUST fill in the correct run_test and prob functions for each problem
# as they are defined in your `interface_function.py` or other relevant files.
# The `multip` and `n_features` values should also be correct for each problem.
PROBS_CONFIG = {
    'atomicity_bypass': ('anomaly_bypass', lambda x, **kwargs: (run_test_bypass(x, **kwargs)), lambda x, **kwargs: (prob_our(x, **kwargs)), 10, 30),
    'broken_barrier': ('Barrier', lambda x, **kwargs: (run_test_berrier(x, **kwargs)), lambda x, **kwargs: (prob_our(x, **kwargs)), 10, 30),
    'broken_peterson': ('peterson', lambda x, **kwargs: (run_test_peterson(x, **kwargs)), lambda x, **kwargs: (prob_our(x, **kwargs)), 10, 30),
    'delayed_write': ('testNset', lambda x, **kwargs: (run_test_our(x, **kwargs)), lambda x, **kwargs: (prob_our(x, **kwargs)), 2, 20), # Assuming testNset maps to Delayed Write or similar
    'flagged_deadlock': ('deadlock', lambda x, **kwargs: (run_test_deadlock(x, **kwargs)), lambda x, **kwargs: (prob_our(x, **kwargs)), 10, 50),
    'if_not_while': ('Non_atomic', lambda x, **kwargs: (run_test_non_atomic(x, **kwargs)), lambda x, **kwargs: (prob_our(x, **kwargs)),10, 30),
    'lock_order_inversion': ('lock', lambda x, **kwargs: (run_test_lock(x, **kwargs)), lambda x, **kwargs: (prob_our(x, **kwargs)), 10, 30),
    'lost_signal': ('signal', lambda x, **kwargs: (run_test_signal(x, **kwargs)), lambda x, **kwargs: (prob_our(x, **kwargs)), 10, 30),
    'partial_lock': ('boolean', lambda x, **kwargs: (run_test_boolean(x, **kwargs)), lambda x, **kwargs: (prob_our(x, **kwargs)), 10,30), # Assuming boolean maps to Partial Lock or similar
    'phantom_permit': ('phantom_permit', lambda x, **kwargs: (run_test_permit(x, **kwargs)), lambda x, **kwargs: (prob_our(x, **kwargs)), 10, 30),
    'race_to_wait': ('waiter', lambda x, **kwargs: (run_test_waiter(x, **kwargs)), lambda x, **kwargs: (prob_our(x, **kwargs)), 10, 30), # Assuming waiter maps to Race-To-Wait
    'racy_increment': ('race', lambda x, **kwargs: (run_test_race(x, **kwargs)), lambda x, **kwargs: (prob_our(x, **kwargs)), 1, 30),
    'semaphore_leak': ('semaphore', lambda x, **kwargs: (run_test_semaphore(x, **kwargs)), lambda x, **kwargs: (prob_our(x, **kwargs)), 10, 30),
    'shared_counter': ('complex', lambda x, **kwargs: (run_test_complex(x, **kwargs)), lambda x, **kwargs: (prob_our(x, **kwargs)), 10, 30), # Assuming complex maps to Shared Counter
    'shared_flag': ('condition', lambda x, **kwargs: (run_test_condition(x, **kwargs)), lambda x, **kwargs: (prob_our(x, **kwargs)), 10, 30), # Assuming condition maps to Shared Flag
    'signal_then_wait': ('signal', lambda x, **kwargs: (run_test_signal(x, **kwargs)), lambda x, **kwargs: (prob_our(x, **kwargs)), 10, 30), # If 'signal' is meant to cover Signal-Then-Wait
    'sleeping_guard': ('sleeping', lambda x, **kwargs: (run_test_sleeping(x, **kwargs)), lambda x, **kwargs: (prob_our(x, **kwargs)), 10, 30),
    # Add other problems here following the format:
    # 'document_name': ('internal_name', run_test_func, prob_func, multip, n_features),
}

# This `probs` list will be dynamically populated in run_classifier.py based on arguments.
# For now, it's just an empty list as a placeholder from constants perspective.
probs = []

# List of all available methods
ALL_METHOD_NAMES = ['Ans', 'Classifier', 'MLP', 'BF', 'SA', 'GA']