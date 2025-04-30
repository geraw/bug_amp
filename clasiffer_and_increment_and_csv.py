# Clasiffer and Increment and CSV

from sklearn.ensemble import RandomForestClassifier

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle
import csv


csv_alg_name = ['Classifier', 'BF', 'SA', 'GA']
csv_cases_name = ['testNset', 'boolean', 'condition', 'Barrier', 'semaphore', 'lock', 'math']
csv_title_line = []
csv_val_name = ['best', '5th', '10th']  # Values to store for each algorithm
csv_rows = []
csv_filename = f'{csv_file_path}{csv_file_name}'

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


csv_A_B_name = ['CL', 'BF', 'Diff', 'Rel']
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
    } for test in range(NUM_OF_TESTS)
}

with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)

    for case in storage:
        for i in storage[case]:
            for alg in storage[case][i]:
                if alg in ['Classifier', 'BF']:
                    for val in csv_val_name:
                        csv_title_line.append(f'{case}_{i}k_{alg}_{val}')
                else:
                    csv_title_line.append(f'{case}_{i}k_{alg}_best')  # Only 'best' for other algorithms
    csv_writer.writerow(csv_title_line)

    for case in storage_ab[0]:
        for i in storage_ab[0][case]:
            # for alg in storage_ab[case][i]:
                for val in csv_A_B_name:  # Include 'best', '5th', '10th' for TT & BF
                    csv_ab_title_line.append(f'{case}_{i*N_TRAIN}k_{val}')
                csv_ab_title_line.append(f'{case}_{i*N_TRAIN}_diff')
                csv_ab_title_line.append(f'{case}_{i*N_TRAIN}_rel')



    for l in range(NUM_OF_TESTS):

        # for name, pr , m, n, lower_bound, upper_bound in probs:
        for name, run_test, prob , multip, n_features in probs:
            print(f'{name=}')

            clf = MLPClassifier(hidden_layer_sizes=(100,  ), activation='relu',
                                random_state=random_state, max_iter=2000 )
                                # random_state=random_state, max_iter=2000, alpha=0.0001 )

            #cls = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
            X_accumulated = np.array([])
            y_accumulated = np.array([])

            correlation_history = []  # Store correlation values
            epsilon = 5.0 #Define epsilon
            cost = 0
            conseq = 3
            m_correletion = 0 # Initialize correlation
            s_correletion = 0 # Initialize correlation
            # clf = MLPClassifier(random_state=1, max_iter=500) #Initialize Classifier
            # ---------------------------------
            # while m_correletion < M_CORRELETION_THRESHOLD and s_correletion < S_CORRELETION_THRESHOLD:
            for _ in range (NUM_TO_CHECK):
                cost += 1
                clf = train(clf)
                print("-----------------")
                # m_correletion, s_correletion = compute_correlations(clf)
                # correlation_history.append(s_correletion)
                print(f"Round {cost}")
                print("---------------------------\n\n")


                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                file_name = f'{file_path}modle_{timestamp}_{cost}_{name}_{file_data}'

                # # Save the trained model
                # with open(file_name, 'wb') as file:
                #     pickle.dump(clf, file)

                _, _, top_reals, _, _, max_real_classifier, X =  find_max_predicted_prob(clf, 15)

                sorted_max_real = sorted(top_reals, reverse=True)
                storage[name][cost]['Classifier']['best'] = sorted_max_real[0] if len(sorted_max_real) > 0 else None
                storage[name][cost]['Classifier']['5th'] = sorted_max_real[4] if len(sorted_max_real) > 4 else None
                storage[name][cost]['Classifier']['10th'] = sorted_max_real[9] if len(sorted_max_real) > 9 else None

                B=30
                _, top_probs, _, _, max_BF =  find_max_prob(15, X)
                print(f'{max_BF=}')
                print(f'\n\n\n')
                storage[name][cost]['BF']['best'] = top_probs[0] if len(top_probs) > 0 else None
                storage[name][cost]['BF']['5th'] = top_probs[4] if len(top_probs) > 4 else None
                storage[name][cost]['BF']['10th'] = top_probs[9] if len(top_probs) > 9 else None



                storage_ab[l][name][cost]['CL'] = sorted_max_real[0]
                storage_ab[l][name][cost]['BF'] = top_probs[0]
                storage_ab[l][name][cost]['Diff'] = storage_ab[l][name][cost]['CL'] - storage_ab[l][name][cost]['BF']
                storage_ab[l][name][cost]['Rel'] = storage_ab[l][name][cost]['CL'] / storage_ab[l][name][cost]['BF'] if storage_ab[l][name][cost]['BF']!=0 else None


                # csv_title_line.append(f'{name}_{cost*1000}_FB')


                bounds = [(0, multip) for _ in range(n_features)]
                D0 = np.array(np.random.rand(n_features)) * multip
                start_time = time.time()
                u_max, pr_max = using_next_point(D0, bounds=bounds, epsilon=multip/20,  k=MAX_TRIALS, iter=int((N_TRAIN*cost)/MAX_TRIALS))
                end_time = time.time()
                pr_max = prob(u_max)
                print(f"\tEitan's convergens: The result for max prob using our method with box of {multip/20} 'prob function' {N_TRAIN*cost} and # runs {count} tests: {pr_max} runtime {(end_time - start_time):.2f}")
                # print(f'{u_max=}')
                storage[name][cost]['SA']['best'] = pr_max


              #-------------------- GA -------------------------------
                # count=0
                # count_ex = 0
                # start_time = time.time()
                # # u_max, pr_max = run_ga(pop_size=max(int(sqrt(NO_FOUND)),15), max_gen=int(ITER/5), bounds=bounds)
                # u_max, pr_max = run_ga(pop_size=50, max_gen=int((N_TRAIN*cost)/50), bounds=bounds)
                # end_time = time.time()
                # pr_max = prob(u_max)
                # print(f"\n\tGenetic Algoritm: The result for max prob using GA method 'pr function' {N_TRAIN*cost} and # runs {count} tests: {pr_max} runtime {(end_time - start_time):.2f}")
                # # print(f'{u_max=}')
                storage[name][cost]['GA']['best'] = 0
                # storage[name][cost]['GA']['best'] = pr_max

                print("\n------------\n\n\n")

        csv_row = []

        for case in csv_cases_name:
            for i in range(1, NUM_TO_CHECK + 1):
                for alg in csv_alg_name:
                    if alg in ['Classifier', 'BF']:  # Include 'best', '5th', '10th' for classifier & BF
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
        csv_ab_rows.append(csv_ab_row)



with open(csv_ab_filename, 'w', newline='') as csvfile:
    csv_ab_writer = csv.writer(csvfile)
    csv_ab_writer.writerow(csv_ab_title_line)
    csv_ab_writer.writerows(csv_ab_rows)


    from scipy.stats import ttest_rel

    all_cl_values = []
    all_bf_values = []
    csv_ab_rows = []
    csv_ab_title_line = []

    csv_ab_title_line.append('Case')
    for i in range(1, NUM_TO_CHECK + 1):
        csv_ab_title_line.append(f'{i*N_TRAIN}_statistic')
        csv_ab_title_line.append(f'{i*N_TRAIN}_p_value')
        csv_ab_title_line.append(f'{i*N_TRAIN}_significantly')

    csv_ab_writer.writerow(csv_ab_title_line)

    for case in csv_cases_name:
        cl_values = []
        bf_values = []
        csv_ab_row = []
        csv_ab_row.append(case)
        # Perform paired t-tests for each index
        for i in range(1, NUM_TO_CHECK + 1):

            for l in range(NUM_OF_TESTS):
                # Ensure the case and index exist in storage_ab
                cl = storage_ab[l][case][i]['CL']
                bf = storage_ab[l][case][i]['BF']
                cl_values.append(cl)
                bf_values.append(bf)
                all_cl_values.append(cl)
                all_bf_values.append(bf)

            # Check if we have enough data to perform the t-test
            if len(cl_values) >= 2:
                # Perform a paired t-test
                statistic, p_value = ttest_rel(cl_values, bf_values)
                csv_ab_row.append(statistic)
                csv_ab_row.append(p_value)
                # Analyze results
                print(f"\nIndex {i}k:")
                print(f"t-statistic: {statistic:.4f}, p-value: {p_value:.4f}")
                if p_value < 0.05:
                    if statistic > 0:
                        csv_ab_row.append('CL significantly than BF')
                        print("V CL performs significantly better than BF.")
                    else:
                        csv_ab_row.append('BF significantly than CL')
                        print("X BF performs significantly better than CL.")
                else:
                    csv_ab_row.append('No statistically')
                    print("No statistically significant difference between CL and BF.")
            else:
                csv_ab_row.append('Not enough data')
                print(f"\nIndex {i}: Not enough data to perform t-test.")

        csv_ab_rows.append(csv_ab_row)
        csv_ab_writer.writerow(csv_ab_row)

    csv_ab_row = []
    csv_ab_row.append('Total')
    # Perform the paired t-test if sufficient data is available
    if len(all_cl_values) >= 2:
        statistic, p_value = ttest_rel(all_cl_values, all_bf_values)
        csv_ab_row.append(statistic)
        csv_ab_row.append(p_value)
        print("\nOverall Comparison:")
        print(f"t-statistic: {statistic:.4f}, p-value: {p_value:.4f}")
        if p_value < 0.05:
            if statistic > 0:
                csv_ab_row.append('CL significantly than BF')
                print("V CL performs significantly better than BF.")
            else:
                csv_ab_row.append('BF significantly than CL')
                print("X BF performs significantly better than CL.")
        else:
            csv_ab_row.append('No statistically')
            print("No statistically significant difference between CL and BF.")
    else:
        csv_ab_row.append('Not enough data')
        print(f"\nIndex {i}: Not enough data to perform t-test.")
    csv_ab_rows.append(csv_ab_row)
    csv_ab_writer.writerow(csv_ab_row)





# prompt: i want to transfer an optiona parameter X for function, and check, if the parameter is transfer then use the array. if not then init it in the function.

def my_function(param1, param2, X=None):
    if X is not None:
        # Use the provided array X
        print("Using provided array X:")
        print(X)
        # Perform operations with X
    else:
        # Initialize X within the function
        print("Initializing array X within the function:")
        X = [1, 2, 3, 4, 5] # Example initialization
        print(X)
        # Perform operations with the initialized X

    # Rest of the function code...
    print("param1:", param1)
    print("param2:", param2)


# prompt: plot clf.predict_proba(X) as a function of two variables and avarage over some random samples of the others

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_predicted_proba(clf):
    #global n_informative, probability_factor, n_features, random_state

    # Create grid for the two variables of interest
    x1 = np.linspace(0, 1, 30)
    x2 = np.linspace(0, 1, 30)
    X1, X2 = np.meshgrid(x1, x2)

    # Initialize an array to store averaged probabilities
    Z = np.zeros_like(X1)

    for i in range(30):
        for j in range(30):
          # Generate random samples for other features
          other_features = rng.rand(1000, n_features - 2)

          # Combine the grid points and the other features
          X_test = np.zeros((1000, n_features))
          X_test[:, 0] = X1[i,j]
          X_test[:, 1] = X2[i,j]
          X_test[:, 2:] = other_features

          # Get the predicted probabilities
          predicted_probs = clf.predict_proba(X_test)[:, 1]

          # Average the probabilities
          Z[i,j] = np.mean(predicted_probs)

    # Create the 3D plot
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Z)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Averaged Predicted Probability')
    ax.set_title('Predicted Probability as a Function of x1 and x2')
    plt.show()

plot_predicted_proba(clf)


# prompt: i want to build a quiry on this list of files that query can be algorithm is our, or query using more then 1 parameter or even quary on the time (between time to time or latest)

import os
import re
import datetime

def parse_filename(filename):
  """Parses a filename and extracts parameters."""
  match = re.match(r"modle_(\d{8})_(\d{6})_(.+?)_(\d+\.\d+)_(\d+\.\d+)_(\d+)_clf_model\.pkl", filename)
  if match:
    date = match.group(1)
    time = match.group(2)
    algorithm = match.group(3)
    m_correlation = float(match.group(4))
    s_correlation = float(match.group(5))
    random_state = int(match.group(6))
    return {
        "date": date,
        "time": time,
        "algorithm": algorithm,
        "m_correlation": m_correlation,
        "s_correlation": s_correlation,
        "random_state": random_state
    }
  else:
    return None

def find_files(directory, algorithm=None, min_m_correlation=None, max_m_correlation=None,
               min_s_correlation=None, max_s_correlation=None, start_date=None, end_date=None):
    """Finds files in a directory matching specific criteria."""
    matching_files = []

    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            parameters = parse_filename(filename)
            if parameters:
                # Check Algorithm
                if algorithm and parameters["algorithm"] != algorithm:
                    continue

                # Check M Correlation
                if min_m_correlation is not None and parameters["m_correlation"] < min_m_correlation:
                    continue
                if max_m_correlation is not None and parameters["m_correlation"] > max_m_correlation:
                    continue

                #Check S correlation
                if min_s_correlation is not None and parameters["s_correlation"] < min_s_correlation:
                    continue
                if max_s_correlation is not None and parameters["s_correlation"] > max_s_correlation:
                    continue

                # Check Date Range
                file_date = datetime.datetime.strptime(parameters["date"], "%Y%m%d").date()
                if start_date and file_date < start_date:
                    continue
                if end_date and file_date > end_date:
                    continue

                matching_files.append(filename)

    return matching_files



# def find_files(directory, algorithm=None, min_m_correlation=None, max_m_correlation=None,
#                min_s_correlation=None, max_s_correlation=None, start_date=None, end_date=None):

directory_path = "/content/"  # Replace with your directory
# # 1. Find all files for the "our" algorithm:
# matching_files = find_files(directory_path, algorithm="our")
# # 2. Find files with m_correlation between 0.2 and 0.6
# matching_files = find_files(directory_path, min_m_correlation=0.25, max_m_correlation=0.5)
# # 3. Find files within a date range:
# start_date = datetime.date(2024, 3, 10)
# end_date = datetime.date(2025, 3, 20)
# matching_files = find_files(directory_path, start_date=start_date, end_date=end_date)
# # 4. Combined Query:
# matching_files = find_files(directory_path, algorithm="our", min_m_correlation = 0.3, start_date=start_date, end_date=end_date)

matching_files = find_files(directory_path, min_m_correlation=0.25, max_m_correlation=0.25, min_s_correlation=0.5, max_s_correlation=0.5, algorithm='our')
for filename in matching_files:
    # Load the saved model
    file_path = os.path.join(directory_path, filename)
    with open(file_path, 'rb') as file:
        loaded_clf = pickle.load(file)

        for name, run_test, prob , multip, n_features in probs:
            print(f'{name=}')
            find_max_predicted_prob(loaded_clf)
            print(f'\n\n\n')
            find_max_prob()


