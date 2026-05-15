---
id: e8e58afd7b
question: Trying to unpickle estimator from version 1.1.1 when using version 0.24.2
sort_order: 31
---


When executing the commands:

```bash
pipenv shell
pipenv run gunicorn --bind 0.0.0.0:9696 predict:app
```

the following warning may occur:

```python
UserWarning: Trying to unpickle estimator DictVectorizer from version 1.1.1 when using version 0.24.2. This might lead to breaking code or invalid results. Use at your own risk.
```

- Ensure you create the virtual environment with the same version of Scikit-Learn that was used to train the model, in this case, version 1.1.1.
- Resolve version conflicts by verifying that the model and `DictVectorizer` files are compatible with the Scikit-Learn version used for the project.