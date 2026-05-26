---
id: 8a8f00802a
question: 'Python_version and Python_full_version error after running pipenv install:'
sort_order: 32
---

If you install packages via `pipenv install`, and encounter an error like this:

```python
pipenv.vendor.plette.models.base.ValidationError: {'python_version': '3.9', 'python_full_version': '3.9.13'}

python_full_version: 'python_version' must not be present with 'python_full_version'

python_version: 'python_full_version' must not be present with 'python_version'
```

Follow these steps to resolve the issue:

1. Open the `Pipfile` in a text editor, such as `nano`:
   
   ```bash
   nano Pipfile
   ```

2. Remove either the `python_version` or `python_full_version` line.

3. Save the changes by pressing `CTRL+X`, then type `Y` and press `Enter`.

4. Run the following command to create the `Pipfile.lock`:

   ```bash
   pipenv lock
   ```

You can now continue with your work.