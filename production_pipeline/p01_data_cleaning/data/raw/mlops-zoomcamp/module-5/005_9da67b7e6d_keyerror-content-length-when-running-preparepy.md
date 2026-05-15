---
id: 9da67b7e6d
question: KeyError ‘content-length’ when running prepare.py
sort_order: 5
---

**Problem:** When running `prepare.py`, encountering `KeyError: 'content-length'`.

**Solution:**

From Emeli Dral: It seems the link used in `prepare.py` to download taxi data is no longer functional. Replace the URL in the script as follows:

```python
url = f"https://nyc-tlc.s3.amazonaws.com/trip+data/{file}"
```

with:

```python
url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{file}"
```

By making this substitution in `prepare.py`, the problem should be resolved, allowing access to the necessary data.