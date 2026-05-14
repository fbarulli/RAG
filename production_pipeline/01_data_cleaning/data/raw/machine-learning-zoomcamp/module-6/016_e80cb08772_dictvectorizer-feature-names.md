---
id: e80cb08772
question: 'DictVectorizer: feature names'
sort_order: 16
---

The DictVectorizer has a function to get the feature names using [`get_feature_names_out()`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html#sklearn.feature_extraction.DictVectorizer.get_feature_names_out:~:text=always%202%2Dd.-,get_feature_names_out,-(input_features%3D)). This is helpful if you need to analyze feature importance but are using the dict vectorizer for one-hot encoding. 

Keep in mind that it returns a NumPy array, so you may need to convert this to a list depending on your usage. For example:

- `dv.get_feature_names_out()` will return an ndarray of string objects.
- `list(dv.get_feature_names_out())` will convert it to a standard list of strings.

Also, ensure that you fit the predictor and response arrays before accessing the feature names.