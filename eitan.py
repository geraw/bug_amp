# Eitan

import numpy as np
import constants
import setup 

def generate_within_bounds(u, epsilon, bounds):
    new_value = [0]*len(u)
    for i in range(len(u)):
        new_value[i] = u[i] + epsilon * np.random.randn()
        while not (bounds[i][0] <= new_value[i] <= bounds[i][1]):
            new_value[i] = u[i] + epsilon * np.random.randn()
    return new_value

def check_valid(u, bounds):
    for i in range(len(u)):
      if not (bounds[i][0] <= u[i] <= bounds[i][1]):
        return False
    return True

# # Generate S with values within the specified bounds
# S = [generate_within_bounds(u, epsilon, bounds) for _ in range(k)]


def next_point(u, epsilon=0.1, k=30, bounds=constants.bounds):
    # # Step 2: Randomly choose k points in the ball B(u, epsilon)
    # S = [u + epsilon * np.random.randn(len(u)) for _ in range(k)]
    # Generate S with values within the specified bounds
    S = [generate_within_bounds(u, epsilon, bounds) for _ in range(k)]


    # Step 3: Execute each x_i and determine whether the bug was found
    id_x = [setup.run_test(np.array(x_i)) for x_i in S]

    # Step 4: Create two averages N and P
    N = np.mean([x_i for x_i, id in zip(S, id_x) if id == 0], axis=0)
    P = np.mean([x_i for x_i, id in zip(S, id_x) if id == 1], axis=0)

    if not isinstance(P, np.ndarray) or not isinstance(N, np.ndarray):
      u_next =  S[0]
    else:
      # Step 5: Obtain a new point w' and take the average of P and w' as the next point in the search
      w_prime = 2*np.array(u) - N
      u_next = (P + w_prime) / 2

      if not check_valid(u_next, bounds):
        u_next = S[0]

    return u_next
#-------------------------------------------

def using_next_point(D0, epsilon=0.1, k=30, iter=1_000, bounds=constants.bounds):

      # Initialize u with a random point
      u = D0
      pr_u=setup.prob(D0)
      # Execute the next_point function and collect cosine similarities
      pr_values = []
      cosine_similarity_values = []
      u_values = []
      max_pr = 0
      max_u = D0

      for _ in range(iter):
          v=u
          pr_v=pr_u
          u = next_point(u, k=k, epsilon=epsilon, bounds=bounds)
          pr_u=setup.run_test(u)
          # if pr_v > pr_u:
          #     u = v
          #     pr_u = pr_v
          if pr_u >= max_pr:
             max_pr = pr_u
             max_u = u

      return max_u, max_pr


# pip install nevergrad

def objective_function(X):
    # Call your existing logic here
    result = setup.prob(X, max_trials=constants.B, no_found=constants.B)  # Example: Replace with your actual function
    return result  # Negate if you want to maximize instead of minimize

