# Learning When Greedy Fails for 0/1 Knapsack

This project tests whether classical ML can predict when ratio-greedy will fail on a 0/1 knapsack instance.

It compares:

- **Greedy:** fast, but not always optimal.
- **Dynamic programming (DP):** exact, but more expensive as capacity grows.
- **ML-guided hybrid:** predicts greedy failure and runs DP only when the predicted probability exceeds a threshold.

DP provides the training label:

```text
failure = greedy_value < optimal_value
```

The dataset contains independent, strongly correlated, weakly correlated, and similar-ratio synthetic instances. Splits are grouped by random seed to avoid leakage between related samples.

## Results

- Logistic regression reached **91% recall** for greedy failures on its
  600-instance validation split.
- At threshold 0.5, the hybrid returned optimal solutions for **93.33%** of the hybrid notebook's validation cases while sending **85.17%** to DP.
- On fresh 40-item instances, it was optimal in **89.90%** of cases and used DP in **82.80%**.
- Lowering the threshold to 0.25 raised the 40-item optimal-solution rate to **99.80%**, but also raised DP usage to **99.70%**.
- Random forest did not outperform logistic regression on solution quality.

The hybrid did **not** produce a runtime improvement on these small synthetic instances: feature extraction and per-instance prediction cost more than the DP work they avoided. The result is therefore a measurable quality-versus-cost trade-off, not a demonstrated speedup.

## Reproduce

```bash
uv sync
uv run jupyter nbconvert --execute --to notebook --inplace notebooks/*.ipynb
uv run --with pytest pytest
```

## Notebook order

1. [`compare_greedy_and_dp.ipynb`](notebooks/compare_greedy_and_dp.ipynb) —
   verify the solvers and observe greedy failures.
2. [`independent_data_analysis.ipynb`](notebooks/independent_data_analysis.ipynb)
   — analyze the initial independent instances.
3. [`other_families_analysis.ipynb`](notebooks/other_families_analysis.ipynb) —
   compare all four instance families.
4. [`classifier.ipynb`](notebooks/classifier.ipynb) — train and evaluate logistic
   regression.
5. [`compare_hybrid.ipynb`](notebooks/compare_hybrid.ipynb) — compare greedy,
   hybrid, and DP.
6. [`distribution_shift.ipynb`](notebooks/distribution_shift.ipynb) — evaluate
   larger instances, thresholds, and random forest.

## Scope

The experiments use only four synthetic families with 20-item training data
and a 40-item size-shift test. They do not establish performance on real-world
data, unseen families, or larger workloads.
