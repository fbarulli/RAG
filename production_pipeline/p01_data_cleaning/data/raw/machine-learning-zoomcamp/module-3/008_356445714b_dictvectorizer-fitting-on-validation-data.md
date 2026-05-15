---
id: 356445714b
question: 'DictVectorizer: Fitting on validation data'
sort_order: 8
---

Validation datasets are used to optimize models by providing an estimate of performance on unseen data. Understanding how to properly use the `DictVectorizer` class is crucial for maintaining this separation between training and validation.

- **Fitting on Training Data**: The `fit` method of `DictVectorizer` analyzes the training dataset to determine how to map dictionary values. Categorical features are one-hot encoded, while numeric features remain unchanged.
- **Avoid Fitting on Validation Data**: Applying the `fit` method to validation data can lead to information leakage, as it exposes the model to data it should not see during training.
- **Appropriate Usage**:
  1. Use `fit_transform` on the training dataset.
  2. Use `transform` only on validation and test datasets.

By following these practices, the model's performance on new data can be more accurately assessed.