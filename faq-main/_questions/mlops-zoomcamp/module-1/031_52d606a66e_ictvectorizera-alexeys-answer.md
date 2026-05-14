---
id: 52d606a66e
question: 'ictVectorizerA: Alexey’s answer'
sort_order: 31
---

In summary:

- `pd.get_dummies` or One-Hot Encoding (OHE) can produce results in different orders and handle missing data differently, potentially causing train and validation sets to have different columns.
- `DictVectorizer` will ignore missing values (during training) and new values (during validation) in datasets.

Other sources:

- [Data Science Stack Exchange](https://datascience.stackexchange.com/questions/9443/when-to-use-one-hot-encoding-vs-labelencoder-vs-dictvectorizor)
- [Scikit-learn Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html)
- [Alteryx: Encode Smarter](https://innovation.alteryx.com/encode-smarter/)

<{IMAGE:image_id}>
