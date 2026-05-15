---
id: af2a35cdf2
question: 'Error: Failed to lock files with Pipfile.lock'
sort_order: 11
---

When adding libraries to the virtual environment in lesson 5.5, the trainer used the command:

```bash
pipenv install numpy scikit-learn==0.24.2 flask
```

However, some people using Python 3.11 or later may encounter an error, failing to lock files correctly with `Pipfile.lock`. You may need to install `scikit-learn==1.4.2` as the error differs from the trainer's example. This should resolve the issue.

If you are still having problems, try the following steps:

- Delete the `Pipfile.lock` using:
  ```bash
  rm Pipfile.lock
  ```
- Rebuild the lock with:
  ```bash
  pipenv lock
  ```
- If it still doesn't work, delete the pipenv environment, `Pipfile`, and `Pipfile.lock`, and create a new one:
  ```bash
  pipenv --rm
  rm Pipfile*
  ```