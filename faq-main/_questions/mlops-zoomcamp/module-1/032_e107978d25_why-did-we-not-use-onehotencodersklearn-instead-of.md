---
id: e107978d25
question: Why did we not use OneHotEncoder(sklearn) instead of DictVectorizer?
sort_order: 32
---

There are several reasons for choosing DictVectorizer over OneHotEncoder:

- **Simple One-Step Process**: DictVectorizer provides a straightforward method to encode both categorical and numerical features from dictionaries, outputting directly to a sparse matrix.
- **Ideal for ML Pipelines**: The direct output in sparse matrix format makes DictVectorizer a good fit for machine learning pipelines without needing additional preprocessing.
- **Use Cases**: 
  - Use **OneHotEncoder** if you need full control, are working with sklearn pipelines, or need to handle unknown categories safely.
  - Use **DictVectorizer** when your data is in dictionary format (e.g., JSON or from APIs) and you aim for quick integration into the pipeline.