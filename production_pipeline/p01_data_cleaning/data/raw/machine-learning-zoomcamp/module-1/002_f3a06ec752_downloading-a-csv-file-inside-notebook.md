---
id: f3a06ec752
question: Downloading a csv file inside notebook
sort_order: 2
---

The best way is to use pandas and give it the URL directly:

```python
url = 'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv'
df = pd.read_csv(url)
```

You can also execute cmd/bash commands inside Jupyter:

```bash
!wget https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv
```

The exclamation mark `!` lets you execute shell commands inside your notebooks. This works for shell commands such as `ls`, `cp`, `mkdir`, `mv`, etc.

For instance, if you then want to move your data into a data directory alongside your notebook-containing directory, you could execute the following:

```bash
!mkdir -p ../data/
!mv housing.csv ../data/
```