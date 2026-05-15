---
id: 0a4f9065b1
question: Loading the dataset directly through Kaggle Notebooks
sort_order: 6
---

To load a dataset in Kaggle Notebooks, you can use the following command. Remember that the `!` before `wget` is essential.

```bash
!wget https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv
```

Once the dataset is loaded onto the Kaggle Notebook server, it can be read using the following pandas command:

```python
df = pd.read_csv('housing.csv')
```