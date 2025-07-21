import numpy as np
import sys
import random

from interface_function import *

M_CORRELETION_THRESHOLD = 0.10    # Magalit correlation thrashhold
S_CORRELETION_THRESHOLD = 0.20    # Spearman correlation threshhold
n_informative = 7
probability_factor = 1
n_features = 50
cost = 0
N_TEST = 1_000                     # Number of random elements for test the coore;etion
count = 0
count_pr = 0
cost = 0
random_state = 42
multip = 1
B=30
S=40
TOP_IN_MARGALITS_CORRELATION = 100
bounds = [(0, multip) for _ in range(n_features)]
MAX_TRIALS = 30
N_INITIAL_SAMPLES = 6
END = 10_000_000

rng = np.random.RandomState(random_state)
# file_path = f'C:\\work\\temp\\BA\\'
# file_path = f'/home/weissye/BugAmplification/BA-exp/BA/'

# drive.mount('/content/drive')
# file_path = f'/content/drive/MyDrive/BugAmplofication2025/results/'
file_path = f'reports/'

NUM_TO_CHECK = 20
NUM_OF_TESTS = 50          # number of tests for calculate the AVR and SDT of the methods
N_TRAIN = 100                   # Number of random elements (increments) that the classifier is trained
N_PARALLEL =1

# Access individual arguments
if len(sys.argv) > 1:
    res_ver = sys.argv[1]
else:
    res_ver = random.randint(0, 1000)

# csv_file_path = f'C:\\work\\temp\\BA\\'
# csv_file_path = f'/home/weissye/BugAmplification/BA-exp/BA/'
# csv_file_path = f'/content/drive/MyDrive/BugAmplication2025/results/'
csv_file_path = f'reports/'
csv_file_name = f'results_{res_ver}.csv'
csv_ab_file_name = f'results_ab_{res_ver}.csv'


file_data = f'{M_CORRELETION_THRESHOLD}_{S_CORRELETION_THRESHOLD}_{random_state}_clf_model.pkl'  # Replace with your desired path
probs = [
# ('Non_atomic', lambda x, **kwargs: (run_test_non_atomic(x, **kwargs)), lambda x, **kwargs: (prob_our(x, **kwargs)),10, 30),
# ('testNset', lambda x, **kwargs: (run_test_our(x, **kwargs)), lambda x, **kwargs: (prob_our(x, **kwargs)),2, 20),
# ('boolean', lambda x, **kwargs: (run_test_boolean(x, **kwargs)), lambda x, **kwargs: (prob_our(x, **kwargs)), 10,30),
# ('condition', lambda x, **kwargs: (run_test_condition(x, **kwargs)), lambda x, **kwargs: (prob_our(x, **kwargs)), 10, 30),
# ('Barrier', lambda x, **kwargs: (run_test_berrier(x, **kwargs)), lambda x, **kwargs: (prob_our(x, **kwargs)), 10, 30),
# ('semaphore', lambda x, **kwargs: (run_test_semaphore(x, **kwargs)), lambda x, **kwargs: (prob_our(x, **kwargs)), 10, 30),
# ('lock', lambda x, **kwargs: (run_test_lock(x, **kwargs)), lambda x, **kwargs: (prob_our(x, **kwargs)), 10, 30),
# ('complex', lambda x, **kwargs: (run_test_complex(x, **kwargs)), lambda x, **kwargs: (prob_our(x, **kwargs)), 10, 30),
# ('dragons', lambda x, **kwargs: (run_test_dragons(x, **kwargs)), lambda x, **kwargs: (prob_our(x, **kwargs)), 10, 30),
# ('peterson', lambda x, **kwargs: (run_test_peterson(x, **kwargs)), lambda x, **kwargs: (prob_our(x, **kwargs)), 10, 30),
# ('deadlock', lambda x, **kwargs: (run_test_deadlock(x, **kwargs)), lambda x, **kwargs: (prob_our(x, **kwargs)), 10, 50),
# ('waiter', lambda x, **kwargs: (run_test_waiter(x, **kwargs)), lambda x, **kwargs: (prob_our(x, **kwargs)), 10, 30),
# ('race', lambda x, **kwargs: (run_test_race(x, **kwargs)), lambda x, **kwargs: (prob_our(x, **kwargs)), 1, 30),
('signal', lambda x, **kwargs: (run_test_signal(x, **kwargs)), lambda x, **kwargs: (prob_our(x, **kwargs)), 10, 30),
# ('sleeping', lambda x, **kwargs: (run_test_sleeping(x, **kwargs)), lambda x, **kwargs: (prob_our(x, **kwargs)), 10, 30),
# ('phantom_permit', lambda x, **kwargs: (run_test_permit(x, **kwargs)), lambda x, **kwargs: (prob_our(x, **kwargs)), 10, 30),
# ('anomaly_bypass', lambda x, **kwargs: (run_test_bypass(x, **kwargs)), lambda x, **kwargs: (prob_our(x, **kwargs)), 10, 30),

]



