---
id: c928a85179
question: 'FutureWarning: Function get_feature_names is deprecated; get_feature_names
  is deprecated in 1.0 and will be removed in 1.2'
sort_order: 9
---

In newer versions of scikit-learn, the method has been replaced by `get_feature_names_out()`.

Instead, use the method `.get_feature_names_out()` from the `DictVectorizer` function to resolve the warning.

```python
# Example usage
from sklearn.feature_extraction import DictVectorizer

# Initialize the vectorizer
vectorizer = DictVectorizer()

# After fitting the vectorizer
vectorizer.fit_transform(...)

# Get feature names
feature_names = vectorizer.get_feature_names_out()
```

Note: The warning indicates that `get_feature_names` will be removed, so switching to `get_feature_names_out` is recommended even though the warning itself won't cause issues yet.