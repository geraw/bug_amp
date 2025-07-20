# Run Classifer

import datetime
import numpy as np
import csv
import time
import argparse
import os # For creating directories if needed

# Import all necessary components from new_constants and other modules
import new_constants
from new_constants import (
    # These will now be defaults, so they don't need to be imported directly here
    # NUM_TO_CHECK, NUM_OF_TESTS, N_TRAIN,
    N_PARALLEL, random_state,
    file_path, file_data, M_CORRELETION_THRESHOLD, S_CORRELETION_THRESHOLD,
    PROBS_CONFIG, ALL_PROBLEM_NAMES_LIST, ALL_METHOD_NAMES
)

# You need to ensure these imports are correct and available in your environment
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC # Not explicitly used, but in original imports
from scipy.stats import ttest_rel # Not explicitly used, but in original imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from scipy.optimize import basinhopping # Not explicitly used in the final version of SA, but in original imports

# Assuming these are available as per your original script
# Make sure eitan.py, ga.py, setup.py, simulate.py are in your Python path
# or same directory.
from eitan import using_next_point # Assuming this is the core function from eitan.py
from ga import run_ga # Assuming this is the core function from ga.py
from setup import generate_initial_examples, accumulate_data, train, find_max_predicted_prob, find_max_prob, set_run_test, set_prob # Assuming these are from setup.py
import simulate # Assuming simulate contains the prob functions if not in interface_function

# --- Global variables (mostly for compatibility with original script structure) ---
single_run_test = None
X_accumulated_clf = []
y_accumulated_clf = []
X_accumulated_sm = []
y_accumulated_sm = []
X_accumulated_mlp = []
y_accumulated_mlp = []
count = 0 # Used in GA and SA for experiment counting

# --- Helper functions (from original script) ---
def run_test_parallel(X, max_trials=1, no_found=1):
    res = run_test_yield(X, max_trials=max_trials, no_found=no_found)
    return 1 if all(list(res)) else 0

def run_test_yield(X, max_trials=1, no_found=1):
    global single_run_test
    # Assuming n_features and new_constants.multip are set correctly for the current problem
    # These are dynamically set within the main loop based on the problem.
    current_n_features = new_constants.n_features # This relies on new_constants.n_features being updated
    for i in range(new_constants.N_PARALLEL):
        x = X[i * current_n_features : (i + 1) * current_n_features]
        yield single_run_test(x, max_trials=max_trials, no_found=no_found)

def run_classifier():
    global count, single_run_test, X_accumulated_clf, y_accumulated_clf, X_accumulated_sm, y_accumulated_sm, \
           X_accumulated_mlp, y_accumulated_mlp

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run bug amplification benchmark.")
    parser.add_argument('--problems', type=str, default='all',
                        help=f"Problems to run. 'all' for all problems, or a comma-separated list of specific problem names (e.g., 'if_not_while,broken_barrier'). Available: {', '.join(ALL_PROBLEM_NAMES_LIST)}")
    parser.add_argument('--method', type=str, default='all',
                        help=f"Methods to run. 'all' for all methods, or a comma-separated list of specific method names (e.g., 'Ens,GA'). Available: {', '.join(ALL_METHOD_NAMES)}")
    parser.add_argument('--NUM_TO_CHECK', type=int, default=new_constants.NUM_TO_CHECK,
                        help=f"Number of increment steps (test budget) for every problem. Default: {new_constants.NUM_TO_CHECK}")
    parser.add_argument('--NUM_OF_TESTS', type=int, default=new_constants.NUM_OF_TESTS,
                        help=f"Number of tests for calculating the AVR and SDT of the methods. Default: {new_constants.NUM_OF_TESTS}")
    parser.add_argument('--N_TRAIN', type=int, default=new_constants.N_TRAIN,
                        help=f"Number of random elements (increments) that the classifier is trained on. Default: {new_constants.N_TRAIN}")
    args = parser.parse_args()

    selected_problems_input = [p.strip() for p in args.problems.split(',')]
    selected_methods_input = [m.strip() for m in args.method.split(',')]

    # Filter problems based on input
    if 'all' in selected_problems_input:
        problems_to_run = ALL_PROBLEM_NAMES_LIST
    else:
        problems_to_run = [p for p in selected_problems_input if p in ALL_PROBLEM_NAMES_LIST]
        if not problems_to_run:
            print(f"Error: No valid problems specified. Choose from: {', '.join(ALL_PROBLEM_NAMES_LIST)}")
            return

    # Filter methods based on input
    if 'all' in selected_methods_input:
        # Exclude 'Classifier' and 'MLP' from ALL_METHOD_NAMES
        methods_to_run = [m for m in ALL_METHOD_NAMES if m not in ['Classifier', 'MLP']]
    else:
        # Filter selected methods, excluding 'Classifier' and 'MLP' if they were explicitly requested
        methods_to_run = [m for m in selected_methods_input if m in ALL_METHOD_NAMES and m not in ['Classifier', 'MLP']]
        if not methods_to_run:
            print(f"Error: No valid methods specified. Choose from: {', '.join([m for m in ALL_METHOD_NAMES if m not in ['Classifier', 'MLP']])}")
            return

    # --- CSV File Naming Adaptation ---
    # Construct dynamic CSV file names based on selected problems and methods
    problem_suffix = 'all' if 'all' in selected_problems_input else '_'.join(sorted(problems_to_run))
    method_suffix = 'all' if 'all' in selected_methods_input else '_'.join(sorted(methods_to_run))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_filename = os.path.join(new_constants.csv_file_path, f'results_{problem_suffix}_{method_suffix}_{timestamp}.csv')
    csv_ab_filename = os.path.join(new_constants.csv_file_path, f'results_ab_{problem_suffix}_{method_suffix}_{timestamp}.csv')

    # Ensure the reports directory exists
    os.makedirs(new_constants.csv_file_path, exist_ok=True)

    # --- CSV Data Structures (Adapted) ---
    csv_alg_name = [m for m in ALL_METHOD_NAMES if m not in ['Classifier', 'MLP']] # Use the filtered list for internal logic
    csv_val_name = ['best', '5th', '10th']

    # Dynamically build csv_cases_name from selected problems
    csv_cases_name = [name for name in problems_to_run]

    # Initialize storage only for selected cases and algorithms
    storage = {
        case: {
            i: {
                alg: {
                    val: None for val in csv_val_name # Initialize with None
                } for alg in csv_alg_name # Use filtered list here
            } for i in range(1, args.NUM_TO_CHECK + 1) # Use args.NUM_TO_CHECK
        } for case in csv_cases_name
    }

    csv_A_B_name = ['Ens', 'BF'] # These are specific comparisons, removed 'CL' and 'MLP'
    storage_ab = {
        test: {
            case: {
                i: {
                    val: None for val in csv_A_B_name
                } for i in range(1, args.NUM_TO_CHECK + 1) # Use args.NUM_TO_CHECK
            } for case in csv_cases_name
        } for test in range(args.NUM_OF_TESTS) # Use args.NUM_OF_TESTS
    }

    csv_rows = []
    csv_ab_rows = []


    with open(csv_filename, 'w', newline='') as csvfile, \
         open(csv_ab_filename, 'w', newline='') as csv_ab_file: # Open both files
        csv_writer = csv.writer(csvfile)
        csv_ab_writer = csv.writer(csv_ab_file)

        # --- Write CSV Headers ---
        csv_title_line = []
        for case in csv_cases_name:
            for i in range(1, args.NUM_TO_CHECK + 1): # Use args.NUM_TO_CHECK
                for alg in methods_to_run: # Only include headers for selected methods
                    if alg in ['Ens', 'BF']: # Only these retain 5th, 10th
                        for val in csv_val_name:
                            csv_title_line.append(f'{case}_{i}k_{alg}_{val}')
                    else: # Only 'best' for other algorithms (SA, GA)
                        csv_title_line.append(f'{case}_{i}k_{alg}_best')
        csv_writer.writerow(csv_title_line)

        csv_ab_title_line = [] # Header for ab_file
        for case in csv_cases_name:
            for i in range(1, args.NUM_TO_CHECK + 1): # Use args.NUM_TO_CHECK
                for alg_comp in csv_A_B_name: # These are fixed comparison values
                    if alg_comp == 'Ens':
                        if 'Ens' in methods_to_run:
                            csv_ab_title_line.append(f'{case}_{i}k_{alg_comp}')
                    elif alg_comp == 'BF':
                        if 'BF' in methods_to_run:
                            csv_ab_title_line.append(f'{case}_{i}k_{alg_comp}')
        csv_ab_writer.writerow(csv_ab_title_line)


        for l in range(args.NUM_OF_TESTS): # Outer loop for number of tests/repetitions, use args.NUM_OF_TESTS
            print(f"--- Running Test Repetition {l+1}/{args.NUM_OF_TESTS} ---") # Use args.NUM_OF_TESTS

            # Initialize problem-specific new_constants at the start of each problem iteration
            # This ensures that n_features and multip are correct for the current problem.
            # Make a local list of problems for this iteration to get functions
            current_probs_list = [PROBS_CONFIG[name] for name in problems_to_run]

            for original_name, run_test_func, prob_func, current_multip, current_n_features in current_probs_list:
                problem_display_name = original_name # Use the original name for display and storage keys
                new_constants.multip = current_multip
                new_constants.n_features = current_n_features
                new_constants.bounds = [(0, new_constants.multip) for _ in range(new_constants.n_features)] # Update bounds

                single_run_test = run_test_func
                set_run_test(run_test_parallel)
                set_prob(prob_func) # Set the global prob function for the modules

                print(f'\nRunning Problem: {problem_display_name}')

                # Reset accumulated data for each new problem
                # Only keep those relevant to the remaining methods (Ens)
                X_accumulated_sm = []
                y_accumulated_sm = []
                # Ensure initial examples are generated for the ensemble
                X_accumulated_sm, y_accumulated_sm = generate_initial_examples()


                # Initialize classifiers/models for each problem iteration
                # Ensure random_state is used for reproducibility
                # Removed mlpclf and clf initializations
                base_learners = [
                    ('lr', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=new_constants.random_state)),
                    ('dt', DecisionTreeClassifier(class_weight='balanced', random_state=new_constants.random_state)),
                    ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=new_constants.random_state)),
                    ('mlp', MLPClassifier(hidden_layer_sizes=(50, 20), activation='relu', solver='adam', alpha=1e-4, learning_rate='adaptive', max_iter=500, early_stopping=True, validation_fraction=0.1, random_state=new_constants.random_state))
                ]
                meta_learner = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=new_constants.random_state)
                stacked_model = StackingClassifier(
                    estimators=base_learners, final_estimator=meta_learner, cv=5, passthrough=True
                )

                new_constants.cost = 0 # Reset cost for each problem

                for _ in range(args.NUM_TO_CHECK): # Inner loop for accumulation/iterations, use args.NUM_TO_CHECK
                    new_constants.cost += 1
                    print(f"Round {l+1} - {problem_display_name} - Iteration {new_constants.cost}/{args.NUM_TO_CHECK}") # Use args.NUM_TO_CHECK
                    print("---------------------------\n")

                    # --- Ensemble (Ens) ---
                    if 'Ens' in methods_to_run:
                        X_accumulated_sm, y_accumulated_sm = accumulate_data(stacked_model, X_accumulated_sm, y_accumulated_sm)
                        X_train, X_test, y_train, y_test = train_test_split(X_accumulated_sm, y_accumulated_sm, stratify=y_accumulated_sm, test_size=0.01, random_state=new_constants.random_state)
                        smote = SMOTE(random_state=new_constants.random_state)
                        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
                        stacked_model.fit(X_train_bal, y_train_bal)
                        predicted_probs = stacked_model.predict_proba(X_accumulated_sm)[:, 1]
                        top_n_indices = np.argsort(predicted_probs)[-15:]
                        top_vectors = [X_accumulated_sm[i] for i in top_n_indices]
                        top_reals = [prob_func(x, max_trials=1000, no_found=1000) for x in top_vectors]
                        sm_sorted_max_real = sorted(top_reals, reverse=True)

                        storage[problem_display_name][new_constants.cost]['Ens']['best'] = sm_sorted_max_real[0] if len(sm_sorted_max_real) > 0 else None
                        storage[problem_display_name][new_constants.cost]['Ens']['5th'] = sm_sorted_max_real[4] if len(sm_sorted_max_real) > 4 else None
                        storage[problem_display_name][new_constants.cost]['Ens']['10th'] = sm_sorted_max_real[9] if len(sm_sorted_max_real) > 9 else None
                        # print(f'\tBest stacked_model - {storage[problem_display_name][new_constants.cost]['Ens']['best']}')
                        # print(f'\t5th best stacked_model - {storage[problem_display_name][new_constants.cost]['Ens']['5th']}')
                        # print(f'\t10th best stacked_model - {storage[problem_display_name][new_constants.cost]['Ens']['10th']}')

                    # Removed Classifier (RandomForest) block
                    # Removed MLP Classifier block

                    # --- BF (Brute Force) ---
                    if 'BF' in methods_to_run:
                        _, top_probs, _, _, max_BF = find_max_prob(n=10) # n=10, so 10 values
                        storage[problem_display_name][new_constants.cost]['BF']['best'] = top_probs[0] if len(top_probs) > 0 else None
                        storage[problem_display_name][new_constants.cost]['BF']['5th'] = top_probs[4] if len(top_probs) > 4 else None
                        storage[problem_display_name][new_constants.cost]['BF']['10th'] = top_probs[9] if len(top_probs) > 9 else None
                        # print(f'\tBest BF - {storage[problem_display_name][new_constants.cost]['BF']['best']}')
                        # print(f'\t5th best BF - {storage[problem_display_name][new_constants.cost]['BF']['5th']}')
                        # print(f'\t10th best BF - {storage[problem_display_name][new_constants.cost]['BF']['10th']}')


                    # --- SA (Simulated Annealing / Eitan's method) ---
                    if 'SA' in methods_to_run:
                        D0 = np.array(np.random.rand(new_constants.n_features)) * new_constants.multip
                        start_time = time.time()
                        u_max, pr_max = using_next_point(D0, bounds=new_constants.bounds, epsilon=new_constants.multip/10,  k=new_constants.MAX_TRIALS, iter=int((2*args.N_TRAIN*new_constants.cost)/new_constants.MAX_TRIALS)) # Use args.N_TRAIN
                        end_time = time.time()
                        pr_max = prob_func(u_max, max_trials=1000, no_found=1000)
                        storage[problem_display_name][new_constants.cost]['SA']['best'] = pr_max
                        print(f"\tEitan's convergens: {pr_max} runtime {(end_time - start_time):.2f}")


                    # --- GA (Genetic Algorithm) ---
                    if 'GA' in methods_to_run:
                        start_time = time.time()
                        u_max, pr_max = run_ga(pop_size=50, max_gen=int((2*args.N_TRAIN*new_constants.cost)/50), bounds=new_constants.bounds) # Use args.N_TRAIN
                        end_time = time.time()
                        pr_max = prob_func(u_max, max_trials=1000, no_found=1000)
                        storage[problem_display_name][new_constants.cost]['GA']['best'] = pr_max
                        print(f"\tGenetic Algoritm: {pr_max} runtime {(end_time - start_time):.2f}")

                    # --- Populate storage_ab for comparison CSV ---
                    if 'Ens' in methods_to_run:
                        storage_ab[l][problem_display_name][new_constants.cost]['Ens'] = storage[problem_display_name][new_constants.cost]['Ens']['best']
                    # Removed 'CL' and 'MLP' population
                    if 'BF' in methods_to_run:
                        storage_ab[l][problem_display_name][new_constants.cost]['BF'] = storage[problem_display_name][new_constants.cost]['BF']['best']

                    # Removed calculation of Diff and Rel as they depended on 'CL' and 'BF'
                    # If you need comparison between Ens and BF, you'd add similar logic here
                    # Example if you wanted Diff/Rel between Ens and BF:
                    # if 'Ens' in methods_to_run and 'BF' in methods_to_run:
                    #     ans_val = storage_ab[l][problem_display_name][new_constants.cost]['Ens']
                    #     bf_val = storage_ab[l][problem_display_name][new_constants.cost]['BF']
                    #     if ans_val is not None and bf_val is not None:
                    #         storage_ab[l][problem_display_name][new_constants.cost]['Diff'] = ans_val - bf_val
                    #         storage_ab[l][problem_display_name][new_constants.cost]['Rel'] = ans_val / bf_val if bf_val != 0 else None


            # --- Write row to main CSV for current repetition (l) ---
            csv_row = []
            for case in csv_cases_name: # Iterate over problems that were run
                for i in range(1, args.NUM_TO_CHECK + 1): # Use args.NUM_TO_CHECK
                    for alg in methods_to_run: # Iterate over methods that were run
                        if alg in ['Ens', 'BF']: # Only these retain 5th, 10th
                            for val in csv_val_name:
                                csv_row.append(storage[case][i][alg][val])
                        else: # SA, GA
                            csv_row.append(storage[case][i][alg]['best'])
            print(f'Main CSV row: {csv_row=}')
            csv_writer.writerow(csv_row)
            csvfile.flush() # Ensure data is written to disk immediately

            # --- Write row to AB CSV for current repetition (l) ---
            csv_ab_row = []
            for case in csv_cases_name:
                for i in range(1, args.NUM_TO_CHECK + 1): # Use args.NUM_TO_CHECK
                    for alg_comp in csv_A_B_name: # This now only includes 'Ens', 'BF'
                        if alg_comp == 'Ens' and 'Ens' in methods_to_run:
                             csv_ab_row.append(storage_ab[l][case][i]['Ens'])
                        elif alg_comp == 'BF' and 'BF' in methods_to_run:
                            csv_ab_row.append(storage_ab[l][case][i]['BF'])
                        # Removed 'CL', 'MLP', 'Diff', 'Rel'
            print(f'AB CSV row: {csv_ab_row=}')
            csv_ab_writer.writerow(csv_ab_row)
            csv_ab_file.flush() # Ensure data is written to disk immediately

if __name__ == "__main__":
    run_classifier()