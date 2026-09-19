Here are the steps for this experiment:
1. **Understand and verify the algorithms [DONE]**
   - Manually solve 3–5 small knapsack instances.
   - Include at least one case where ratio-greedy fails.
   - Implement ratio-greedy.
   - Implement the full 2D dynamic-programming algorithm yourself.
   - Test both against your manual answers.
2. **Build the experimental foundation [DONE]**
   - Define an `Item(weight, value)` representation.
   - Make both solvers return the chosen items and total value.
   - Add automated tests for feasibility, expected values, empty inputs, and items heavier than capacity.
   - Only proceed once DP reliably supplies the ground truth.
3. **Generate and inspect data [DONE]**
   - Start with one family: independently sampled weights and values.
   - Generate perhaps 1,000 instances.
   - Record greedy value, optimal value, failure label, and relative gap.
   - Plot failure frequency and gap distribution before doing any ML.
4. **Expand instance families**
   - Add correlated, similar-ratio, and deliberately adversarial instances.
   - Keep a `family` column so you can evaluate each group separately.
   - Split using generation seeds, not arbitrary rows from one generated batch.
5. **Train the first classifier**
   - Add scikit-learn only at this stage.
   - Begin with logistic regression.
   - Examine recall for greedy failures, not merely accuracy.
   - Investigate which features appear useful.
6. **Construct the actual hybrid**
   - If predicted failure probability exceeds a threshold, run DP.
   - Otherwise, accept greedy.
   - Compare always-greedy, hybrid, and always-DP using:
     - optimal-solution rate,
     - mean and maximum gap,
     - percentage sent to DP,
     - end-to-end runtime,
     - false negatives.
7. **Test distribution shift**
   - Evaluate larger instances or an unseen generator family.
   - Vary the classification threshold to plot the quality-versus-DP-usage tradeoff.
   - Only then try a random forest as the nonlinear comparison.