---
id: 744f73a640
question: Different values of AUC, each time code is re-run
sort_order: 15
---

When running `dt = DecisionTreeClassifier()` in Jupyter on the same laptop, different AUC values are observed each time it is re-run or after restarting the kernel. Examples include values like 0.674, 0.652, 0.642, 0.669, etc. This variability is discussed in a video between 7:40-7:45 of section 6.3.

**Solution:**

- Set a random seed to ensure reproducibility by using:

  ```python
  dt = DecisionTreeClassifier(random_state=22)
  ```