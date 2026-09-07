# Learning When Greedy Fails for 0/1 Knapsack

## Experiment description

This experiment studies whether a classical machine-learning model can learn
when a fast greedy algorithm is likely to produce a suboptimal solution to the
0/1 knapsack problem.

Given `n` items, item `i` has weight `w_i` and value `v_i`. The goal is to choose
binary decisions `x_i` that maximize total value while respecting capacity
`W`:

```text
maximize    sum(v_i * x_i)
subject to  sum(w_i * x_i) <= W
            x_i in {0, 1}
```

The experiment compares three approaches:

1. **Greedy solver:** Sort items by value-to-weight ratio and add each item if
   it fits. This method is fast, but it is not guaranteed to be optimal for
   0/1 knapsack.
2. **Dynamic-programming solver:** Compute the exact optimal value. This serves
   as the ground truth, although its running time grows with both the number of
   items and the capacity.
3. **ML-guided hybrid solver:** Use a classifier to predict whether greedy will
   fail on a new instance. If failure is likely, run dynamic programming;
   otherwise, accept the greedy solution.

Dynamic programming will be used offline to label generated training
instances. For an instance, the classifier target is whether the greedy value
is lower than the optimal value:

```text
failure = 1 if greedy_value < optimal_value, otherwise 0
```

The relative optimality gap will measure the severity of a failure:

```text
relative_gap = (optimal_value - greedy_value) / optimal_value
```

Candidate input features include:

- Number of items and knapsack capacity
- Capacity divided by total item weight
- Mean, spread, minimum, and maximum of weights and values
- Statistics of the value-to-weight ratios
- Correlation between item weights and values
- Capacity used and capacity remaining after the greedy solution
- Number and proportion of items selected by greedy

The initial ML comparison will use logistic regression as an interpretable
baseline and either a random forest or a small multilayer perceptron as a more
flexible model.

### Instance generation

Training and evaluation data should contain several instance families:

- Independently sampled weights and values
- Strongly correlated weights and values
- Weakly correlated weights and values
- Items with similar value-to-weight ratios
- Hand-constructed or searched-for greedy counterexamples

All weights should be positive. If zero-weight items are later allowed,
positive-value zero-weight items should be included before ratio-based sorting,
because their value-to-weight ratio is undefined.

The data should be split by random seed and instance family. In addition to an
in-distribution test set, the experiment should include an out-of-distribution
test set with larger item counts, different capacity ranges, or an unseen
instance family.

### Evaluation

The individual predictors will be evaluated with classification metrics such
as precision, recall, F1 score, ROC-AUC, and a confusion matrix. Because the
classifier is part of a solver, the primary evaluation will focus on downstream
behavior:

- Average and maximum relative optimality gap
- Percentage of instances solved optimally
- Percentage of instances sent to dynamic programming
- End-to-end runtime
- Performance for each instance family
- Performance on out-of-distribution instances

The classifier threshold will be varied to produce a tradeoff curve between
dynamic-programming usage and solution quality. The hybrid should be compared
with the two endpoints: always use greedy and always use dynamic programming.

## Learning goals

By completing this experiment, I expect to learn:

1. How to formalize 0/1 knapsack using decision variables, an objective, and a
   capacity constraint.
2. Why a locally sensible greedy rule can fail to find the global optimum.
3. How dynamic programming uses overlapping subproblems to guarantee an exact
   solution.
4. How to quantify approximation quality using absolute and relative
   optimality gaps.
5. How to construct a dataset that represents multiple types of optimization
   instances rather than a single convenient random distribution.
6. How to use exact algorithmic solutions as supervised-learning labels.
7. How to engineer interpretable features that describe the structure and
   difficulty of a knapsack instance.
8. How to distinguish classifier accuracy from the quality of the complete
   decision-making system.
9. How to evaluate generalization under distribution shift.
10. How ML can select between algorithms instead of replacing a reliable
    classical algorithm unnecessarily.

## Expected results

These are hypotheses to test, not results that should be assumed in advance.

- Dynamic programming should solve every generated instance optimally but
  become slower as capacities and item counts increase.
- Greedy should be substantially faster and often optimal on ordinary random
  instances, while failing more frequently on specially structured or
  adversarial instances.
- Logistic regression should identify some broad patterns associated with
  greedy failure, but it may miss nonlinear interactions between items.
- A random forest or small neural network should predict greedy failures more
  accurately in-distribution, although the more complex model may generalize
  less reliably to unseen instance families.
- The ML-guided hybrid should occupy a middle point between the two classical
  baselines: it should call dynamic programming less often than the always-DP
  method while achieving better solution quality than the always-greedy method.
- Raising the classifier threshold should reduce DP usage but increase the risk
  of accepting suboptimal greedy solutions. Lowering it should improve solution
  quality at greater computational cost.
- Performance will probably decline under distribution shift, demonstrating
  that an algorithm selector is only as reliable as the instance distribution
  represented in its training data.

The main question is therefore not whether ML can outperform an exact solver in
solution quality. It is whether ML can preserve near-optimal quality while
reducing how often the exact solver must be used.
