# Bug Amplification Benchmark

This repository provides a comprehensive benchmark suite and search framework for amplifying concurrency bugs in multi-threaded software using delay injection and machine learning-guided heuristics.

## 🚀 Project Overview

Modern concurrent software often suffers from rare and elusive bugs that manifest only under specific timing conditions. This project provides:

- A taxonomy of concurrency bugs categorized by observable effect and root cause.
- A benchmark of 17 problems covering all categories using generator-based simulations.
- A suite of search strategies including Brute Force, Simulated Annealing, Genetic Algorithms, and an Ensemble Stacking Classifier.

## 📁 Repository Structure

```
bug_amp/
├── problems/            # Python generator implementations of benchmark problems
├── framework/           # Search methods and execution environment
├── reports/             # Generated results in CSV and LaTeX
├── utils/               # Helper scripts for visualization and logging
├── run_benchmark.py     # Entry point for experiments
└── requirements.txt     # Python dependencies
```

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

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## 🧾 Citation

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

---

Maintained by [Yeshayahu Weiss](https://github.com/geraw).

