---
id: b78f247dc5
question: 'ValueError: feature_names must be string, and may not contain [, ] or <:'
sort_order: 6
---

When creating DMatrix for train and validation, you might encounter the error:

```
ValueError: feature_names must be string, and may not contain [, ] or <
```

The cause of this error is special characters in feature names, such as `=` and `<`. To fix this error, you can remove or replace these characters:

```python
features = [i.replace("=<", "_").replace("=","_") for i in features]
```


If the equal sign `=` is not a problem for you, the following adjustment could also work:

```python
features = []

for f in dv.feature_names_:
    string = f.replace("=<", "-le")
    features.append(string)
```
