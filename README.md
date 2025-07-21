# Bug Amplification Benchmark

This repository provides a comprehensive benchmark suite and search framework for amplifying concurrency bugs in multi-threaded software using delay injection and machine learning-guided heuristics.

## 🚀 Project Overview

Modern concurrent software often suffers from rare and elusive bugs that manifest only under specific timing conditions. This project provides:

- A taxonomy of concurrency bugs categorized by observable effect and root cause.
- A benchmark of 17 problems covering all categories using generator-based simulations.
- A suite of search strategies including Brute Force, Simulated Annealing, Genetic Algorithms, and an Ensemble Stacking Classifier.

## 📁 Repository Structure

````markdown
bug_amp/
├── problems/            # Python generator implementations of benchmark problems
├── framework/           # Search methods and execution environment
├── reports/             # Generated results in CSV and LaTeX
├── utils/               # Helper scripts for visualization and logging
├── run_benchmark.py     # Entry point for experiments
└── requirements.txt     # Python dependencies
````

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/geraw/bug_amp.git
cd bug_amp
pip install -r requirements.txt
```

> Requires Python 3.9 or higher.

## 📈 Usage Examples

Run the benchmark across all problems:

```bash
python run_benchmark.py --problems all --method ensemble
```

Run a specific problem with a selected search method:

```bash
python run_benchmark.py --problems if_not_while --method ga
```

**`run_benchmark.py` Parameters:**

The `run_benchmark.py` script now accepts the following additional parameters:

  * `--problems PROBLEMS`: Problems to run. 'all' for all problems, or a comma-separated list of specific problem names (e.g., 'if_not_while,broken_barrier'). Available: atomicity_bypass, broken_barrier, broken_peterson, delayed_write, flagged_deadlock, if_not_while, lock_order_inversion, lost_signal, partial_lock, phantom_permit, race_to_wait, racy_increment, semaphore_leak, shared_counter, shared_flag, signal_then_wait, sleeping_guard.
  * `--method METHOD`: Methods to run. 'all' for all methods, or a comma-separated list of specific method names (e.g., 'Ens,GA'). Available: Ens, BF, SA, GA.
  * `--NUM_TO_CHECK NUM_TO_CHECK`: Number of increment steps (test budget) for every problem. Default: 20.
  * `--NUM_OF_TESTS NUM_OF_TESTS`: Number of tests for calculating the AVR and SDT of the methods. Default: 50.
  * `--N_TRAIN N_TRAIN`: Number of random elements (increments) that the classifier is trained on. Default: 100.

## 🧪 Benchmark Details

Each problem in the `problems/` folder is a synthetic but representative example of a concurrency bug pattern, such as:

  - Non-atomic operations
  - Signal-before-wait
  - Double-unlock deadlocks
  - Conditional races

Each test injects timing distortions using `yield` and noise parameters to simulate thread interleaving.

## 🔍 Search Methods

We support four search strategies:

  - **Brute Force (BF)**: Exhaustive search over the parameter space.
  - **Simulated Annealing (SA)**: Stochastic local search.
  - **Genetic Algorithm (GA)**: Evolutionary optimization.
  - **Ensemble Classifier (ENS)**: Learns from past failures to guide testing.

## 📊 Results & Evaluation

Results are stored in the `reports/` folder and include:

  - Success probability plots
  - Comparison graphs (PNG and LaTeX PGFPlots)
  - Statistical significance tests (Wilcoxon signed-rank)

## 📜 License

This project is licensed under the MIT License – see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

## 📟 Citation

If you use this benchmark or framework, please cite:

```bibtex
@inproceedings{weiss2025amplification,
  title={Amplification of Evasive Concurrency Bugs via Delayed Execution},
  author={Weiss, Yeshayahu and Others},
  booktitle={Proceedings of the ICSE},
  year={2025}
}
```

## 🤝 Contributing

Pull requests are welcome! Please open an issue to discuss changes.

-----

Maintained by [Yeshayahu Weiss](https://github.com/geraw).
