---
id: a2d10b83d9
question: 'Pipenv installation not working (AttributeError: module ''collections''
  has no attribute ''MutableMapping'')'
sort_order: 4
---

If you encounter pipenv failures with the command `pipenv install` and see the following error:

```python
AttributeError: module 'collections' has no attribute 'MutableMapping'
```

The issue occurs because you are using the system Python (3.10) for pipenv.

To resolve this issue:

1. If pipenv was previously installed via `apt-get`, remove it using:
   ```bash
   sudo apt remove pipenv
   ```

2. Ensure a non-system Python is installed in your environment. An easy way to achieve this is by installing Anaconda or Miniconda.

3. Install pipenv using your non-system Python:
   ```bash
   pip install pipenv
   ```

4. Re-run `pipenv install <dependencies>` with the relevant dependencies. It should work without issues.

This solution was tested and worked on an AWS instance similar to the configuration presented in class.