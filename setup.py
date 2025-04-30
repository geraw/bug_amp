from typing import Callable
from sklearn.exceptions import NotFittedError

from interface_function import *
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau

from constants import *
from sklearn.neural_network import MLPClassifier
from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement
from sklearn.utils import shuffle as util_shuffle
from sklearn.datasets import make_classification
from sklearn.calibration import CalibratedClassifierCV


from sklearn.model_selection import train_test_split


# Add global variable declaration
run_test: Callable = None

def set_run_test(test_function: Callable) -> None:
    global run_test
    run_test = test_function


def sample_best_Xs_using_model(clf, n=1):
    global run_test

    X_candidates = rng.rand(N_TRAIN*1_000, n_features) * multip

    predicted_probs = clf.predict_proba(X_candidates)[:, 1]
    top_n_indices = np.argsort(predicted_probs)[-N_TRAIN:]

    X_in = [X_candidates[i] for i in top_n_indices]
    y_in = np.array([run_test(x) for x in X_in])

    print(f"{len(y_in)=}  {sum(y_in)/len(y_in)=}")
    return X_in, y_in


def generate_test_data():
    global run_test
  
    X_in = rng.rand(N_TRAIN, n_features) * multip
    y_in = np.array([run_test(x) for x in X_in])
    print(f"{len(y_in)=}  {sum(y_in)/len(y_in)=}")
    return X_in, y_in


def accumulate_data(clf):
    """Accumulates training data, potentially using a classifier for sampling.

    Args:
        clf: A classifier object (optional). If provided, it's used to sample
            the best Xs using `sample_best_Xs_using_model`.

    Returns:
        A tuple containing the accumulated X and y data as NumPy arrays.
    """
    global X_accumulated, y_accumulated

    try:
      # Exploration
      X, y = generate_test_data()
      X_accumulated = np.concatenate([X_accumulated, X])
      y_accumulated = np.concatenate([y_accumulated, y])

      # Exploitation
      X, y = sample_best_Xs_using_model(clf)
      X_accumulated = np.concatenate([X_accumulated, X])
      y_accumulated = np.concatenate([y_accumulated, y])
    except NotFittedError:
      pass

    return X_accumulated, y_accumulated

def train(clf):
    # global random_state, n_informative, probability_factor, n_features
    X, y = accumulate_data(clf)

    print( f"{len(y)=}  {sum(y)/len(y)=}")
    return clf.fit(X, y)


import pandas as pd

def display_results(real_probs, predicted_probs):
  # Create a pandas DataFrame
  data = {'real': real_probs,
          'pred': predicted_probs}
  df = pd.DataFrame(data)

  # Display the DataFrame as a table
  display(df)


# from minepy import MINE

def compute_correlations(clf):
    # global random_state, n_informative, probability_factor, n_features, N_TEST

    # Create random X
    X = rng.rand(N_TEST, n_features) * multip
    # X = rng.rand(N_TEST, n_features) * 10


    # Get the predicted probabilities for the positive class (second column)
    predicted_probs = clf.predict_proba(X)[:, 1]


    # # Calculate prob(X) for each test sample
    # ##########################################
    real_probs = np.array([prob(x) for x in X]) ### should be prob with x times

    # max__real_index = np.argmax(real_probs)
    # max_predict_index = np.argmax(predicted_probs)

    # print(f"{real_probs[max__real_index]=}")
    # print(f"{predicted_probs[max_predict_index]=}")

    # # Show the results in a table
    # display_results(real_probs, predicted_probs)

    # Compute the Pearson correlation coefficient
    # correlation = pearsonr(predicted_probs, real_probs)[0]

    # #===================================
    # Calculate Margalit's correlation
    df = pd.DataFrame({'X': list(X), 'probability': predicted_probs})
    df = df.sort_values('probability', ascending=False).head(TOP_IN_MARGALITS_CORRELATION)

    m_correlation = sum([run_test(X) for X in df['X']])/TOP_IN_MARGALITS_CORRELATION
    # #===================================

    print(f"Margalit's rank correlation: {m_correlation}")

    # print(f"Pearson correlation coefficient: {correlation}")

    # Calculate Spearman's rank correlation
    # #################
    s_correlation, p_value = spearmanr(predicted_probs, real_probs)

    print(f"Spearman's rank correlation: {s_correlation}")
    # #################

    return m_correlation, s_correlation


import random

def get_n_random_items(D, n):
  """Gets n random items from array D.

  Args:
    D: The input array.
    n: The number of random items to retrieve.

  Returns:
    A list containing n random items from D, or all items if n is
    larger than the length of D.
  """
  if n >= len(D):
    return D[:]  # Return a copy of the entire array if n is too large
  else:
    # Convert D to a list of lists if it's a multidimensional array
    if D.ndim > 1:
        D = D.tolist()
    # if type(D) is not list:
    #   D = D.tolist() # added a check if X is not a list
    return random.sample(D, n)

from pickle import TRUE
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau

from scipy.optimize import minimize


def func_to_minimize(X):
  return 1 - clf.predict_proba([X])[0,1]


# ===============================================================================
def find_max_prob(n=1, D=None ):
    if D is None:
        X = rng.rand(int((cost*N_TRAIN)/B), n_features) * multip
    else:
        X = get_n_random_items(D, int((cost*N_TRAIN)/B))
    probs = [prob(x, max_trials=B, no_found=B) for x in X]

    # Get indices of top n probabilities
    top_n_indices = np.argsort(probs)[-n:][::-1]  # Reverse order

    top_vectors = [X[i] for i in top_n_indices]
    top_probs = [probs[i] for i in top_n_indices]

    # Calculate average and std
    avg_prob = np.mean(top_probs)
    std_prob = np.std(top_probs)

    # print(f"Top {n} vectors: {top_vectors}")
    # print(f"Top {n} probabilities: {top_probs}")
    # print(f"Average probability: {avg_prob}")
    # print(f"Standard deviation of probabilities: {std_prob}")

    return top_vectors, top_probs, avg_prob, std_prob, top_probs[0]


######################################################################
# A_B_array = [[cls(random(100)), bf(random(100))] for i in range(1000)]

# A_B_array = [[cls(x:=random(100)), bf(x)] for i in range(1000)]

# Is there statistically significant advantage to A over B
# No







# prompt: i want that functions 'fine_max_predicted_prob' recives another parameter  called 'n' with default is 1, finds the best 'n ' resuls in the minimize and return the list of top n vectors, the top prob, the averege and the STD of the n top results

def find_max_predicted_prob(clf, n=1):
    #global random_state, n_informative, probability_factor, n_features

    X_candidates = rng.rand(N_TEST, n_features) * multip # Generate a larger set of candidates

    predicted_probs = clf.predict_proba(X_candidates)[:, 1]

    top_n_indices = np.argsort(predicted_probs)[-n:]
    top_vectors = [X_candidates[i] for i in top_n_indices]
    top_probs = [predicted_probs[i] for i in top_n_indices]
    top_reals = [prob(x) for x in top_vectors]

    sorted_max_real = sorted(top_reals, reverse=True)

    return top_vectors, top_probs, sorted_max_real, sorted_max_real[0], X_candidates



