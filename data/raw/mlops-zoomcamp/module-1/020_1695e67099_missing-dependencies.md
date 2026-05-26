---
id: 1695e67099
images:
- description: 'image #1'
  id: image_1
  path: images/mlops-zoomcamp/image_7c6ef087.png
question: Missing dependencies
sort_order: 20
---

If some dependencies are missing:

<{IMAGE:image_1}>

Install the following packages:

- `pandas`
- `matplotlib`
- `scikit-learn`
- `fastparquet`
- `pyarrow`
- `seaborn`

```bash
pip install -r requirements.txt
```

I have seen this error when using `pandas.read_parquet()`. The solution is to install `pyarrow` or `fastparquet` by running the following command in the notebook:

```bash
!pip install pyarrow
```

**Note:** If you’re using Conda instead of pip, install `fastparquet` rather than `pyarrow`, as it is much easier to install and it’s functionally identical to `pyarrow` for our needs.