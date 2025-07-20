# Run Classifer

import datetime
import numpy as np
import csv
import time
import argparse
import os # For creating directories if needed

# Import all necessary components from constants and other modules
import constants
from constants import (
    NUM_TO_CHECK, NUM_OF_TESTS, N_TRAIN, N_PARALLEL, random_state,
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
    # Assuming n_features and constants.multip are set correctly for the current problem
    # These are dynamically set within the main loop based on the problem.
    current_n_features = constants.n_features # This relies on constants.n_features being updated
    for i in range(constants.N_PARALLEL):
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
        methods_to_run = ALL_METHOD_NAMES
    else:
        methods_to_run = [m for m in selected_methods_input if m in ALL_METHOD_NAMES]
        if not methods_to_run:
            print(f"Error: No valid methods specified. Choose from: {', '.join(ALL_METHOD_NAMES)}")
            return

    # --- CSV File Naming Adaptation ---
    # Construct dynamic CSV file names based on selected problems and methods
    problem_suffix = 'all' if 'all' in selected_problems_input else '_'.join(sorted(problems_to_run))
    method_suffix = 'all' if 'all' in selected_methods_input else '_'.join(sorted(methods_to_run))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_filename = os.path.join(constants.csv_file_path, f'results_{problem_suffix}_{method_suffix}_{timestamp}.csv')
    csv_ab_filename = os.path.join(constants.csv_file_path, f'results_ab_{problem_suffix}_{method_suffix}_{timestamp}.csv')

    # Ensure the reports directory exists
    os.makedirs(constants.csv_file_path, exist_ok=True)

    # --- CSV Data Structures (Adapted) ---
    csv_alg_name = ALL_METHOD_NAMES # Use the full list for internal logic
    csv_val_name = ['best', '5th', '10th']

    # Dynamically build csv_cases_name from selected problems
    csv_cases_name = [name for name in problems_to_run]

    # Initialize storage only for selected cases and algorithms
    storage = {
        case: {
            i: {
                alg: {
                    val: None for val in csv_val_name # Initialize with None
                } for alg in csv_alg_name
            } for i in range(1, NUM_TO_CHECK + 1)
        } for case in csv_cases_name
    }

    csv_A_B_name = ['Ans', 'CL', 'MLP', 'BF', 'Diff', 'Rel'] # These are specific comparisons
    storage_ab = {
        test: {
            case: {
                i: {
                    val: None for val in csv_A_B_name
                } for i in range(1, NUM_TO_CHECK + 1)
            } for case in csv_cases_name
        } for test in range(constants.NUM_OF_TESTS)
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
            for i in range(1, NUM_TO_CHECK + 1):
                for alg in methods_to_run: # Only include headers for selected methods
                    if alg in ['Ans','Classifier', 'MLP', 'BF']:
                        for val in csv_val_name:
                            csv_title_line.append(f'{case}_{i}k_{alg}_{val}')
                    else: # Only 'best' for other algorithms (SA, GA)
                        csv_title_line.append(f'{case}_{i}k_{alg}_best')
        csv_writer.writerow(csv_title_line)

        csv_ab_title_line = [] # Header for ab_file
        for case in csv_cases_name:
            for i in range(1, NUM_TO_CHECK + 1):
                for alg in csv_A_B_name: # These are fixed comparison values
                    if alg in ['Diff', 'Rel']: # Diff/Rel are comparisons between CL and BF
                        if 'Classifier' in methods_to_run and 'BF' in methods_to_run:
                            csv_ab_title_line.append(f'{case}_{i}k_{alg}')
                    elif alg == 'Ans':
                        if 'Ans' in methods_to_run:
                            csv_ab_title_line.append(f'{case}_{i}k_{alg}')
                    elif alg == 'CL':
                        if 'Classifier' in methods_to_run:
                            csv_ab_title_line.append(f'{case}_{i}k_{alg}')
                    elif alg == 'MLP':
                        if 'MLP' in methods_to_run:
                            csv_ab_title_line.append(f'{case}_{i}k_{alg}')
                    elif alg == 'BF':
                        if 'BF' in methods_to_run:
                            csv_ab_title_line.append(f'{case}_{i}k_{alg}')
        csv_ab_writer.writerow(csv_ab_title_line)


        for l in range(NUM_OF_TESTS): # Outer loop for number of tests/repetitions
            print(f"--- Running Test Repetition {l+1}/{NUM_OF_TESTS} ---")

            # Initialize problem-specific constants at the start of each problem iteration
            # This ensures that n_features and multip are correct for the current problem.
            # Make a local list of problems for this iteration to get functions
            current_probs_list = [PROBS_CONFIG[name] for name in problems_to_run]

            for original_name, run_test_func, prob_func, current_multip, current_n_features in current_probs_list:
                problem_display_name = original_name # Use the original name for display and storage keys
                constants.multip = current_multip
                constants.n_features = current_n_features
                constants.bounds = [(0, constants.multip) for _ in range(constants.n_features)] # Update bounds

                single_run_test = run_test_func
                set_run_test(run_test_parallel)
                set_prob(prob_func) # Set the global prob function for the modules

                print(f'\nRunning Problem: {problem_display_name}')

                # Reset accumulated data for each new problem
                X_accumulated_clf, y_accumulated_clf = generate_initial_examples()
                X_accumulated_sm = list(X_accumulated_clf) # Ensure deep copy if mutable
                y_accumulated_sm = list(y_accumulated_clf)
                X_accumulated_mlp = list(X_accumulated_clf)
                y_accumulated_mlp = list(y_accumulated_clf)


                # Initialize classifiers/models for each problem iteration
                # Ensure random_state is used for reproducibility
                mlpclf = MLPClassifier(
                    hidden_layer_sizes=(50, 20), activation='relu', solver='adam',
                    alpha=1e-4, learning_rate='adaptive', max_iter=500,
                    early_stopping=True, validation_fraction=0.1, random_state=constants.random_state
                )
                clf = RandomForestClassifier(
                    n_estimators=100, max_depth=None, max_features='sqrt',
                    random_state=constants.random_state # Add random_state for RF
                )
                base_learners = [
                    ('lr', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=constants.random_state)),
                    ('dt', DecisionTreeClassifier(class_weight='balanced', random_state=constants.random_state)),
                    ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=constants.random_state)),
                    ('mlp', MLPClassifier(hidden_layer_sizes=(50, 20), activation='relu', solver='adam', alpha=1e-4, learning_rate='adaptive', max_iter=500, early_stopping=True, validation_fraction=0.1, random_state=constants.random_state))
                ]
                meta_learner = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=constants.random_state)
                stacked_model = StackingClassifier(
                    estimators=base_learners, final_estimator=meta_learner, cv=5, passthrough=True
                )

                constants.cost = 0 # Reset cost for each problem

                for _ in range(NUM_TO_CHECK): # Inner loop for accumulation/iterations
                    constants.cost += 1
                    print(f"Round {l+1} - {problem_display_name} - Iteration {constants.cost}/{NUM_TO_CHECK}")
                    print("---------------------------\n")

                    # --- Ensemble (Ans) ---
                    if 'Ans' in methods_to_run:
                        X_accumulated_sm, y_accumulated_sm = accumulate_data(stacked_model, X_accumulated_sm, y_accumulated_sm)
                        X_train, X_test, y_train, y_test = train_test_split(X_accumulated_sm, y_accumulated_sm, stratify=y_accumulated_sm, test_size=0.01, random_state=constants.random_state)
                        smote = SMOTE(random_state=constants.random_state)
                        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
                        stacked_model.fit(X_train_bal, y_train_bal)
                        predicted_probs = stacked_model.predict_proba(X_accumulated_sm)[:, 1]
                        top_n_indices = np.argsort(predicted_probs)[-15:]
                        top_vectors = [X_accumulated_sm[i] for i in top_n_indices]
                        top_reals = [prob_func(x, max_trials=1000, no_found=1000) for x in top_vectors]
                        sm_sorted_max_real = sorted(top_reals, reverse=True)

                        storage[problem_display_name][constants.cost]['Ans']['best'] = sm_sorted_max_real[0] if len(sm_sorted_max_real) > 0 else None
                        storage[problem_display_name][constants.cost]['Ans']['5th'] = sm_sorted_max_real[4] if len(sm_sorted_max_real) > 4 else None
                        storage[problem_display_name][constants.cost]['Ans']['10th'] = sm_sorted_max_real[9] if len(sm_sorted_max_real) > 9 else None
                        print(f'\tBest stacked_model - {storage[problem_display_name][constants.cost]['Ans']['best']}')
                        print(f'\t5th best stacked_model - {storage[problem_display_name][constants.cost]['Ans']['5th']}')
                        print(f'\t10th best stacked_model - {storage[problem_display_name][constants.cost]['Ans']['10th']}')


                    # --- Classifier (RandomForest) ---
                    if 'Classifier' in methods_to_run:
                        X_accumulated_clf, y_accumulated_clf = accumulate_data(clf, X_accumulated_clf, y_accumulated_clf)
                        X_train, X_test, y_train, y_test = train_test_split(X_accumulated_clf, y_accumulated_clf, stratify=y_accumulated_clf, test_size=0.01, random_state=constants.random_state)
                        smote = SMOTE(random_state=constants.random_state)
                        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
                        clf = train(clf, X_train_bal, y_train_bal)
                        predicted_probs_clf = clf.predict_proba(X_accumulated_clf)[:, 1]
                        top_n_indices_clf = np.argsort(predicted_probs_clf)[-15:]
                        top_vectors_clf = [X_accumulated_clf[i] for i in top_n_indices_clf]
                        top_reals_clf = [prob_func(x, max_trials=1000, no_found=1000) for x in top_vectors_clf]
                        sorted_max_real = sorted(top_reals_clf, reverse=True)

                        storage[problem_display_name][constants.cost]['Classifier']['best'] = sorted_max_real[0] if len(sorted_max_real) > 0 else None
                        storage[problem_display_name][constants.cost]['Classifier']['5th'] = sorted_max_real[4] if len(sorted_max_real) > 4 else None
                        storage[problem_display_name][constants.cost]['Classifier']['10th'] = sorted_max_real[9] if len(sorted_max_real) > 9 else None
                        print(f'\tBest Classifier - {storage[problem_display_name][constants.cost]['Classifier']['best']}')
                        print(f'\t5th best Classifier - {storage[problem_display_name][constants.cost]['Classifier']['5th']}')
                        print(f'\t10th best Classifier - {storage[problem_display_name][constants.cost]['Classifier']['10th']}')

                    # --- MLP Classifier ---
                    if 'MLP' in methods_to_run:
                        X_accumulated_mlp, y_accumulated_mlp = accumulate_data(mlpclf, X_accumulated_mlp, y_accumulated_mlp)
                        X_train, X_test, y_train, y_test = train_test_split(X_accumulated_mlp, y_accumulated_mlp, stratify=y_accumulated_mlp, test_size=0.01, random_state=constants.random_state)
                        smote = SMOTE(random_state=constants.random_state)
                        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
                        mlpclf = train(mlpclf, X_train_bal, y_train_bal)
                        predicted_probs_mlp = mlpclf.predict_proba(X_accumulated_mlp)[:, 1]
                        top_n_indices_mlp = np.argsort(predicted_probs_mlp)[-15:]
                        top_vectors_mlp = [X_accumulated_mlp[i] for i in top_n_indices_mlp]
                        top_reals_mlp = [prob_func(x, max_trials=1000, no_found=1000) for x in top_vectors_mlp]
                        mlp_sorted_max_real = sorted(top_reals_mlp, reverse=True)

                        storage[problem_display_name][constants.cost]['MLP']['best'] = mlp_sorted_max_real[0] if len(mlp_sorted_max_real) > 0 else None
                        storage[problem_display_name][constants.cost]['MLP']['5th'] = mlp_sorted_max_real[4] if len(mlp_sorted_max_real) > 4 else None
                        storage[problem_display_name][constants.cost]['MLP']['10th'] = mlp_sorted_max_real[9] if len(mlp_sorted_max_real) > 9 else None
                        print(f'\tBest MLP Classifier - {storage[problem_display_name][constants.cost]['MLP']['best']}')
                        print(f'\t5th best MLP Classifier - {storage[problem_display_name][constants.cost]['MLP']['5th']}')
                        print(f'\t10th best MLP Classifier - {storage[problem_display_name][constants.cost]['MLP']['10th']}')


                    # --- BF (Brute Force) ---
                    if 'BF' in methods_to_run:
                        _, top_probs, _, _, max_BF = find_max_prob(n=10) # n=10, so 10 values
                        storage[problem_display_name][constants.cost]['BF']['best'] = top_probs[0] if len(top_probs) > 0 else None
                        storage[problem_display_name][constants.cost]['BF']['5th'] = top_probs[4] if len(top_probs) > 4 else None
                        storage[problem_display_name][constants.cost]['BF']['10th'] = top_probs[9] if len(top_probs) > 9 else None
                        print(f'\tBest BF - {storage[problem_display_name][constants.cost]['BF']['best']}')
                        print(f'\t5th best BF - {storage[problem_display_name][constants.cost]['BF']['5th']}')
                        print(f'\t10th best BF - {storage[problem_display_name][constants.cost]['BF']['10th']}')


                    # --- SA (Simulated Annealing / Eitan's method) ---
                    if 'SA' in methods_to_run:
                        D0 = np.array(np.random.rand(constants.n_features)) * constants.multip
                        start_time = time.time()
                        u_max, pr_max = using_next_point(D0, bounds=constants.bounds, epsilon=constants.multip/10,  k=constants.MAX_TRIALS, iter=int((2*N_TRAIN*constants.cost)/constants.MAX_TRIALS))
                        end_time = time.time()
                        pr_max = prob_func(u_max, max_trials=1000, no_found=1000)
                        storage[problem_display_name][constants.cost]['SA']['best'] = pr_max
                        print(f"\tEitan's convergens: {pr_max} runtime {(end_time - start_time):.2f}")


                    # --- GA (Genetic Algorithm) ---
                    if 'GA' in methods_to_run:
                        start_time = time.time()
                        u_max, pr_max = run_ga(pop_size=50, max_gen=int((2*N_TRAIN*constants.cost)/50), bounds=constants.bounds)
                        end_time = time.time()
                        pr_max = prob_func(u_max, max_trials=1000, no_found=1000)
                        storage[problem_display_name][constants.cost]['GA']['best'] = pr_max
                        print(f"\tGenetic Algoritm: {pr_max} runtime {(end_time - start_time):.2f}")

                    # --- Populate storage_ab for comparison CSV ---
                    if 'Ans' in methods_to_run:
                        storage_ab[l][problem_display_name][constants.cost]['Ans'] = storage[problem_display_name][constants.cost]['Ans']['best']
                    if 'Classifier' in methods_to_run:
                        storage_ab[l][problem_display_name][constants.cost]['CL'] = storage[problem_display_name][constants.cost]['Classifier']['best']
                    if 'MLP' in methods_to_run:
                        storage_ab[l][problem_display_name][constants.cost]['MLP'] = storage[problem_display_name][constants.cost]['MLP']['best']
                    if 'BF' in methods_to_run:
                        storage_ab[l][problem_display_name][constants.cost]['BF'] = storage[problem_display_name][constants.cost]['BF']['best']

                    # Calculate Diff and Rel only if both CL and BF were run
                    if 'Classifier' in methods_to_run and 'BF' in methods_to_run:
                        cl_val = storage_ab[l][problem_display_name][constants.cost]['CL']
                        bf_val = storage_ab[l][problem_display_name][constants.cost]['BF']
                        if cl_val is not None and bf_val is not None:
                            storage_ab[l][problem_display_name][constants.cost]['Diff'] = cl_val - bf_val
                            storage_ab[l][problem_display_name][constants.cost]['Rel'] = cl_val / bf_val if bf_val != 0 else None


            # --- Write row to main CSV for current repetition (l) ---
            csv_row = []
            for case in csv_cases_name: # Iterate over problems that were run
                for i in range(1, NUM_TO_CHECK + 1):
                    for alg in methods_to_run: # Iterate over methods that were run
                        if alg in ['Ans', 'Classifier', 'MLP', 'BF']:
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
                for i in range(1, NUM_TO_CHECK + 1):
                    for alg_comp in csv_A_B_name:
                        # Only append if the method corresponding to the comparison was run
                        if alg_comp == 'Ans' and 'Ans' in methods_to_run:
                             csv_ab_row.append(storage_ab[l][case][i]['Ans'])
                        elif alg_comp == 'CL' and 'Classifier' in methods_to_run:
                            csv_ab_row.append(storage_ab[l][case][i]['CL'])
                        elif alg_comp == 'MLP' and 'MLP' in methods_to_run:
                             csv_ab_row.append(storage_ab[l][case][i]['MLP'])
                        elif alg_comp == 'BF' and 'BF' in methods_to_run:
                            csv_ab_row.append(storage_ab[l][case][i]['BF'])
                        elif alg_comp in ['Diff', 'Rel'] and 'Classifier' in methods_to_run and 'BF' in methods_to_run:
                            csv_ab_row.append(storage_ab[l][case][i][alg_comp])
                        # If a method was not selected, its corresponding value in the row will be skipped,
                        # ensuring alignment with the header generated for selected methods.
            print(f'AB CSV row: {csv_ab_row=}')
            csv_ab_writer.writerow(csv_ab_row)
            csv_ab_file.flush() # Ensure data is written to disk immediately

if __name__ == "__main__":
    run_classifier()