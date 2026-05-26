---
id: e35e6adc78
question: What is the difference between OneHotEncoder and DictVectorizer?
sort_order: 17
---

Both work in similar ways to convert categorical features to numerical variables for use in training a model. The difference lies in the input:

- **OneHotEncoder** uses an array as input.
- **DictVectorizer** uses a dictionary.

Both will produce the same result. However, with OneHotEncoder, features are sorted alphabetically. With DictVectorizer, you stack features as desired.