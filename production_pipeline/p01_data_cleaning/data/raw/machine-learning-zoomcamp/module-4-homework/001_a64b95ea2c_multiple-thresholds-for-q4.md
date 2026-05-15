---
id: a64b95ea2c
question: Multiple thresholds for Q4
sort_order: 1
---

I am getting multiple thresholds with the same F1 score. Does this indicate I am doing something wrong, or is there a method for choosing? Should I just pick the lowest?

- Choose the threshold closest to any of the options.

You can also use `scikit-learn` (or other standard libraries/packages) to verify results obtained using your own code. For example, use `classification_report` to obtain precision, recall, and F1-score.

Refer to the documentation: [scikit-learn classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)