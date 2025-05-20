# Run Classifer

import datetime
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier


import numpy as np
import csv
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.stats import ttest_rel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

from imblearn.over_sampling import SMOTE
from eitan import *
from ga import *

import constants
from constants import *
from setup import *
import simulate
import time


single_run_test = None
csv_alg_name = ['Ans','Classifier', 'BF', 'SA', 'GA']
csv_title_line = []
csv_val_name = ['best', '5th', '10th']  # Values to store for each algorithm
csv_rows = []
csv_filename = f'{csv_file_path}{csv_file_name}'

# csv_cases_name = ['Non_atomic', 'testNset', 'boolean', 'condition', 'Barrier', 'semaphore', 'lock', 'complex', 'dragons', 'math']
csv_cases_name = []
for name, _, _ , _, _ in probs:
    csv_cases_name.append(name)

# Updated structure: Each algorithm stores three values ('best', '5th', '10th')
storage = {
    case: {
        i: {
            alg: {
                val: 0 for val in csv_val_name
            } for alg in csv_alg_name
        } for i in range(1, NUM_TO_CHECK + 1)
    } for case in csv_cases_name
}

csv_A_B_name = ['Ans', 'CL', 'BF', 'Diff', 'Rel']
csv_ab_title_line = []
csv_ab_rows = []
csv_ab_filename = f'{csv_file_path}{csv_ab_file_name}'

storage_ab = {
    test: {
        case: {
            i: {
                val: 0 for val in csv_A_B_name
            } for i in range(1, NUM_TO_CHECK + 1)
        } for case in csv_cases_name
    } for test in range(constants.NUM_OF_TESTS)
}

all_cl_values = []
all_bf_values = []
csv_ab_rows = []


def run_test_parallel(X, max_trials=1, no_found=1):
    res = run_test_yield(X, max_trials=max_trials, no_found=no_found)
    # Return 1 if all results are True (simulations return 1), else return 0
    return 1 if all(list(res)) else 0

def run_test_yield(X, max_trials=1, no_found=1):
    
    global single_run_test
    for i in range(constants.N_PARALLEL):
        # Extract the slice for the current iteration
        x = X[i*n_features : (i + 1)*n_features]
        # Run run_test and yield the result
        # Run single_run_test with isolated arguments and yield the result
        yield single_run_test(x, max_trials=max_trials, no_found=no_found)  # Simulate returns 0 if no assertion fails
    
    
def run_classifier():
    global count,single_run_test, X_accumulated_clf, y_accumulated_clf, X_accumulated_sm, y_accumulated_sm
    count = 0
    

    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        for case in storage:
            for i in storage[case]:
                for alg in storage[case][i]:
                    if alg in ['Ans','Classifier', 'BF']:
                        for val in csv_val_name:
                            csv_title_line.append(f'{case}_{i}k_{alg}_{val}')
                    else:
                        csv_title_line.append(f'{case}_{i}k_{alg}_best')  # Only 'best' for other algorithms
        
        csv_writer.writerow(csv_title_line)

        for l in range(NUM_OF_TESTS):

            # for name, pr , m, n, lower_bound, upper_bound in probs:
            for name, run_test, prob , constants.multip, constants.n_features in constants.probs:
                
                single_run_test = run_test
                set_run_test(run_test_parallel)
                set_prob(prob)
                
                print(f'{name=}')

                # clf = MLPClassifier(hidden_layer_sizes=(100,  ), activation='relu',
                #                     random_state=random_state, max_iter=2000 )
                                    # random_state=random_state, max_iter=2000, alpha=0.0001 )

                # clf = MLPClassifier(
                #     hidden_layer_sizes=(50, 20),  # 2 hidden layers: 50 → 20 units
                #     activation='relu',           # Good general-purpose activation
                #     solver='adam',               # Robust optimizer, good for most tasks
                #     alpha=1e-4,                  # L2 regularization to avoid overfitting
                #     learning_rate='adaptive',    # Slows learning when not improving
                #     max_iter=500,                # Allow time to converge
                #     early_stopping=True,         # Automatically stop if no improvement
                #     validation_fraction=0.1      # Use 10% of train set for early stopping
                # )

                # clf = CalibratedClassifierCV(clf, method='isotonic')

                clf = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=None,
                    max_features='sqrt',
                )
                
                base_learners = [
                    ('lr', LogisticRegression(max_iter=1000, class_weight='balanced')),
                    ('dt', DecisionTreeClassifier(class_weight='balanced')),
                    ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced')),
                    ('mlp', MLPClassifier(hidden_layer_sizes=(50, 20), activation='relu', solver='adam', alpha=1e-4, learning_rate='adaptive', max_iter=500, early_stopping=True, validation_fraction=0.1, random_state=random_state))
                ]

                meta_learner = LogisticRegression(class_weight='balanced',max_iter=1000)

                stacked_model = StackingClassifier(
                    estimators=base_learners,
                    final_estimator=meta_learner,
                    cv=5,
                    passthrough=True
                )

                #cls = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
                X_accumulated_clf, y_accumulated_clf = generate_initial_examples()
                X_accumulated_sm = X_accumulated_clf
                y_accumulated_sm = y_accumulated_clf
                
                # print (f"Initial data: {len(X_accumulated_clf)=} {len(y_accumulated_clf)=} {len(X_accumulated_sm)=} {len(y_accumulated_sm)=}")

                correlation_history = []  # Store correlation values
                epsilon = 5.0 #Define epsilon
                constants.cost = 0
                conseq = 3
                m_correletion = 0 # Initialize correlation
                s_correletion = 0 # Initialize correlation
                # clf = MLPClassifier(random_state=1, max_iter=500) #Initialize Classifier
                # ---------------------------------
                # while m_correletion < M_CORRELETION_THRESHOLD and s_correletion < S_CORRELETION_THRESHOLD:
                for _ in range (NUM_TO_CHECK):
                    constants.cost += 1

                    #-------------------- Classifier Preperations -------------------------------

                    print("-----------------")
                    # m_correletion, s_correletion = compute_correlations(clf)
                    # correlation_history.append(s_correletion)
                    print(f"Round {l} - {constants.cost}")
                    print("---------------------------\n\n")


                    #-------------------- Ansamble - Classifiers -------------------------------

                    X_accumulated_sm, y_accumulated_sm = accumulate_data(stacked_model, X_accumulated_sm, y_accumulated_sm)
                    # print (f"Ansamble data: {len(X_accumulated_clf)=} {len(y_accumulated_clf)=} {len(X_accumulated_sm)=} {len(y_accumulated_sm)=}")

                    X_train, X_test, y_train, y_test = train_test_split(X_accumulated_sm, y_accumulated_sm, stratify=y_accumulated_sm, test_size=0.01, random_state=random_state)
                    
                    unique, counts = np.unique(y_accumulated_sm, return_counts=True)
                    print(f'SM {dict(zip(unique, counts))=}')  # e.g., {0: 4500, 1: 500}

                    smote = SMOTE(random_state=random_state)
                    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

                    print("After SMOTE:", dict(zip(*np.unique(y_train_bal, return_counts=True))))

                    stacked_model.fit(X_train_bal, y_train_bal)
                    
                    
                    
                    #-----------------------------
                    probs = stacked_model.predict_proba(X_test)

                    bug_probs = probs[:, 1]
                    predicted_probs = stacked_model.predict_proba(X_accumulated_sm)[:, 1]

                    top_buggy_indices = np.argsort(predicted_probs)[::-1][:5]
                    for i in top_buggy_indices:
                        print(f"Input: {X_accumulated_sm[i]}, score X ({prob(X_accumulated_sm[i]):.4f})")  


                    
                    
                    # predicted_probs = stacked_model.predict_proba(X_train_bal)[:, 1]
                    # top_n_indices = np.argsort(predicted_probs)[-15:]
                    # top_vectors = [X_train_bal[i] for i in top_n_indices]
                    # # top_probs = [predicted_probs[i] for i in top_n_indices]
                    # top_reals = [prob(x) for x in top_vectors]

                    # sm_sorted_max_real = sorted(top_reals, reverse=True)
                    # print(f'sm_sorted_max_real = {sm_sorted_max_real}')
                    # print(f'\n')
                    # print(f'\tBest stacked_model balanced - {sm_sorted_max_real[0] if len(sm_sorted_max_real) > 0 else None}')
                    # print(f'\t5th best stacked_model balanced- {sm_sorted_max_real[4] if len(sm_sorted_max_real) > 4 else None}')
                    # print(f'\t10th best stacked_model balanced- {sm_sorted_max_real[9] if len(sm_sorted_max_real) > 9 else None}')

                    # #------------------------------

                    # # print(classification_report(y_test, y_pred, digits=4))
                    
                    # _, _, sm_sorted_max_real, max_real_classifier, X =  find_max_predicted_prob(stacked_model, 15)
                    predicted_probs = stacked_model.predict_proba(X_accumulated_sm)[:, 1]
                    top_n_indices = np.argsort(predicted_probs)[-15:]
                    top_vectors = [X_accumulated_sm[i] for i in top_n_indices]
                    # top_probs = [predicted_probs[i] for i in top_n_indices]
                    top_reals = [prob(x) for x in top_vectors]

                    sm_sorted_max_real = sorted(top_reals, reverse=True)

                    # sorted_max_real = sorted(top_reals, reverse=True)
                    print(f'\n')
                    print(f'\tBest stacked_model - {sm_sorted_max_real[0] if len(sm_sorted_max_real) > 0 else None}')
                    print(f'\t5th best stacked_model - {sm_sorted_max_real[4] if len(sm_sorted_max_real) > 0 else None}')
                    print(f'\t10th best stacked_model - {sm_sorted_max_real[9] if len(sm_sorted_max_real) > 0 else None}')


                    #-------------------- CL - Classifier -------------------------------
        
                    X_accumulated_clf, y_accumulated_clf = accumulate_data(clf, X_accumulated_clf, y_accumulated_clf)
                    # print (f"CL data: {len(X_accumulated_clf)=} {len(y_accumulated_clf)=} {len(X_accumulated_sm)=} {len(y_accumulated_sm)=}")

                    # Check class distribution
                    unique, counts = np.unique(y_accumulated_clf, return_counts=True)
                    print(f'clf {dict(zip(unique, counts))=}')  # e.g., {0: 4500, 1: 500}
                    X_train, X_test, y_train, y_test = train_test_split(X_accumulated_clf, y_accumulated_clf, stratify=y_accumulated_clf, test_size=0.01, random_state=42)

                    smote = SMOTE(random_state=42)
                    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

                    print(f"After SMOTE:, {dict(zip(*np.unique(y_train_bal, return_counts=True)))}")
                    
                    clf = train(clf, X_train_bal, y_train_bal)
                    # _, _, sorted_max_real, max_real_classifier, X =  find_max_predicted_prob(clf, 15)
                    # # sorted_max_real = sorted(top_reals, reverse=True)
                    # print(f'\n')
                    # print(f'\tBest Classifier balanced - {sorted_max_real[0] if len(sorted_max_real) > 0 else None}')
                    # print(f'\t5th best Classifier balanced - {sorted_max_real[4] if len(sorted_max_real) > 4 else None}')
                    # print(f'\t10th best Classifier balanced - {sorted_max_real[9] if len(sorted_max_real) > 9 else None}')

                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    file_name = f'{file_path}modle_{timestamp}_{constants.cost}_{name}_{file_data}'

                    # # Save the trained model
                    # with open(file_name, 'wb') as file:
                    #     pickle.dump(clf, file)
                    # clf = train(clf, X_accumulated_clf, y_accumulated_clf)

                    # _, _, sorted_max_real, max_real_classifier, X =  find_max_predicted_prob(clf, 15)
                    predicted_probs_clf = clf.predict_proba(X_accumulated_clf)[:, 1]
                    top_n_indices_clf = np.argsort(predicted_probs_clf)[-15:]
                    top_vectors_clf = [X_accumulated_clf[i] for i in top_n_indices_clf]
                    # top_probs = [predicted_probs[i] for i in top_n_indices]
                    top_reals_clf = [prob(x) for x in top_vectors_clf]

                    sorted_max_real = sorted(top_reals_clf, reverse=True)

                    # sorted_max_real = sorted(top_reals, reverse=True)
                    print(f'\n')
                    print(f'\tBest Classifier - {sorted_max_real[0] if len(sorted_max_real) > 0 else None}')
                    print(f'\t5th best Classifier - {sorted_max_real[4] if len(sorted_max_real) > 4 else None}')
                    print(f'\t10th best Classifier - {sorted_max_real[9] if len(sorted_max_real) > 9 else None}')


                #-------------------- BF - Brout Force -------------------------------

                    B=30
                    _, top_probs, _, _, max_BF =  find_max_prob(n=10)
                    print(f'\n\n\n')
                    print(f'\tBest BF - {top_probs[0]}') if len(top_probs) > 0 else print('\tBest BF - None')
                    print(f'\t5th best BF - {top_probs[4]}') if len(top_probs) > 4 else print('\t5th best BF - None')
                    print(f'\t10th best BF - {top_probs[9]}')  if len(top_probs) > 9 else print('\t10th best BF - None')


                #-------------------- SA - Eitan -------------------------------

                    bounds = [(0, constants.multip) for _ in range(n_features)]
                    D0 = np.array(np.random.rand(n_features)) * constants.multip
                    start_time = time.time()
                    u_max, pr_max = using_next_point(D0, bounds=bounds, epsilon=multip/10,  k=MAX_TRIALS, iter=int((2*N_TRAIN*constants.cost)/MAX_TRIALS))
                    end_time = time.time()
                    pr_max = prob(u_max)
                    print(f"\n\n\n\tEitan's convergens: The result for max prob using our method with box of {multip/20} 'prob function' {N_TRAIN*constants.cost} and # runs {count} tests: {pr_max} runtime {(end_time - start_time):.2f}")
                    # print(f'{u_max=}')
                    storage[name][constants.cost]['SA']['best'] = pr_max


                #-------------------- GA -------------------------------
                    # count=0
                    count_ex = 0
                    start_time = time.time()
                    # u_max, pr_max = run_ga(pop_size=max(int(sqrt(NO_FOUND)),15), max_gen=int(ITER/5), bounds=bounds)
                    u_max, pr_max = run_ga(pop_size=50, max_gen=int((2*N_TRAIN*constants.cost)/50), bounds=bounds)
                    end_time = time.time()
                    pr_max = prob(u_max)
                    print(f"\n\tGenetic Algoritm: The result for max prob using GA method 'pr function' {N_TRAIN*constants.cost} and # runs {count} tests: {pr_max} runtime {(end_time - start_time):.2f}")
                    # storage[name][constants.cost]['GA']['best'] = 0
                    storage[name][constants.cost]['GA']['best'] = pr_max

                    print("\n------------\n\n\n")

                #-------------------- Save results  -------------------------------

                    storage[name][constants.cost]['Ans']['best'] = sm_sorted_max_real[0] if len(sm_sorted_max_real) > 0 else None
                    storage[name][constants.cost]['Ans']['5th'] = sm_sorted_max_real[4] if len(sm_sorted_max_real) > 4 else None
                    storage[name][constants.cost]['Ans']['10th'] = sm_sorted_max_real[9] if len(sm_sorted_max_real) > 9 else None

                    storage[name][constants.cost]['Classifier']['best'] = sorted_max_real[0] if len(sorted_max_real) > 0 else None
                    storage[name][constants.cost]['Classifier']['5th'] = sorted_max_real[4] if len(sorted_max_real) > 4 else None
                    storage[name][constants.cost]['Classifier']['10th'] = sorted_max_real[9] if len(sorted_max_real) > 9 else None

                    storage[name][constants.cost]['BF']['best'] = top_probs[0] if len(top_probs) > 0 else None
                    storage[name][constants.cost]['BF']['5th'] = top_probs[4] if len(top_probs) > 4 else None
                    storage[name][constants.cost]['BF']['10th'] = top_probs[9] if len(top_probs) > 9 else None

                    storage_ab[l][name][constants.cost]['Ans'] = sm_sorted_max_real[0]
                    storage_ab[l][name][constants.cost]['CL'] = sorted_max_real[0]
                    storage_ab[l][name][constants.cost]['BF'] = top_probs[0]
                    storage_ab[l][name][constants.cost]['Diff'] = storage_ab[l][name][constants.cost]['CL'] - storage_ab[l][name][constants.cost]['BF']
                    storage_ab[l][name][constants.cost]['Rel'] = storage_ab[l][name][constants.cost]['CL'] / storage_ab[l][name][constants.cost]['BF'] if storage_ab[l][name][constants.cost]['BF']!=0 else None

            csv_row = []

            for case in csv_cases_name:
                for i in range(1, NUM_TO_CHECK + 1):
                    for alg in csv_alg_name:
                        if alg in ['Ans', 'Classifier', 'BF']:  # Include 'best', '5th', '10th' for classifier & BF
                            for val in csv_val_name:  # csv_val_name = ['best', '5th', '10th']
                                csv_row.append(storage[case][i][alg][val])
                        else:  # Only 'best' for other algorithms
                            csv_row.append(storage[case][i][alg]['best'])

            print(f'{csv_row=}')
            csv_rows.append(csv_row)
            csv_writer.writerow(csv_row)



            csv_ab_row = []
            for case in csv_cases_name:
                for i in range(1, NUM_TO_CHECK + 1):
                    for alg in csv_A_B_name:
                        csv_ab_row.append(storage_ab[l][case][i][alg])
            print(f'{csv_ab_row=}')
            print("\n------------\n\n\n")
            csv_ab_rows.append(csv_ab_row)

if __name__ == "__main__":
    run_classifier()







# pip install nevergrad

# import nevergrad as ng

# # Define the objective function
# def objective_function(x):
#     # Replace this with your actual logic
#     # Example: minimize the square of x
#     return x[0]**2 + x[1]**2

# # Define the search space
# parametrization = ng.p.Array(shape=(2,)).set_bounds(-10, 10)  # Example: 2D search space with bounds

# # Create the Simulated Annealing optimizer
# optimizer = ng.optimizers.SimulatedAnnealing(parametrization=parametrization, budget=100)

# # Run the optimization
# recommendation = optimizer.minimize(objective_function)

# # Get the best point
# best_point = recommendation.value
# print(f"Best point: {best_point}")


# def objective_function(X):
#     # Call your existing logic here
#     result = setup.run_test(X)  # Example: Replace with your actual function
#     return -result  # Negate if you want to maximize instead of minimize