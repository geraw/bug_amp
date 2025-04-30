# Mathematical Benchmark

def prob_math(X, max_trials=S, no_found=S):   #for generalizing
    x = np.array(X[:n_informative])
    center = 0.5
    return np.exp(-np.sum((x - center)**10) * 10**6)


def run_test_math(X):
  return 1 if rng.rand() < prob(X) else 0


