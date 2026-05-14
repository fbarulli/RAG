---
id: 3f9286329a
question: The feature size is different for training set and validation set
sort_order: 42
---

While working through HW1, you may notice that the feature sizes for the training and validation datasets are different. This issue often arises when using the incorrect method with a dictionary vectorizer.

Ensure you use the `transform` method on the premade dictionary vectorizer instead of `fit_transform`. Since you already have the dictionary vectorizer created, there's no need to execute the fit pipeline on the model.