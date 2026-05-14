---
id: 58a9c117d5
question: Hyperopt is not installable with Conda
sort_order: 48
---

**Description**

When setting up your virtual environment with

```bash
conda install --file requirements.txt
```

you may encounter the following error:

```bash
PackagesNotFoundError: The following packages are not available from current channels:

- hyperopt
```

**Solution**

- Your conda installation might be out of date. You can update Conda with:

  ```bash
  conda update -n base -c defaults conda
  ```

- If updating does not solve the issue, consider installing the package via the Intel channel, as advised on the [conda page](https://anaconda.org/intel/hyperopt):

  ```bash
  conda install intel::hyperopt
  ```