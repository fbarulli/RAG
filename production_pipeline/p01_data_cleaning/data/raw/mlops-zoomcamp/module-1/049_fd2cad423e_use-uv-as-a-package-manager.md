---
id: fd2cad423e
question: Use uv as a package manager
sort_order: 49
---

There is an option to run the project without Anaconda while easily managing multiple Python versions on your machine. The new package manager, `uv`, is a fast and efficient one written in Rust. It's recommended for use in Python projects overall. [Install guide](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv venv --python 3.9.7 # install python 3.9.7 used in the course

source .venv/bin/activate # activate the environment

python -V # should be 3.9.7

uv pip install pandas scikit-learn notebook seaborn pyarrow # install required packages

jupyter notebook # run jupyter notebook
```

Cleanup is straightforward. Deactivate the environment and delete the folder:

```bash
deactivate

rm -rf .venv
```