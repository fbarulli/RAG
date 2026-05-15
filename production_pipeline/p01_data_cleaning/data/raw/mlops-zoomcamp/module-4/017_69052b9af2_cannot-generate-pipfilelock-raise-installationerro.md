---
id: 69052b9af2
question: Cannot generate pipfile.lock raise InstallationError( pip9.exceptions.InstallationError)
sort_order: 17
---

Problem description: Cannot generate `pipfile.lock`. Raises `InstallationError( pip9.exceptions.InstallationError: Command "python setup.py egg_info" failed with error code 1`.

Solution:

- You need to force an upgrade of `wheel` and `pipenv`.
- Run the following command:

  ```bash
  pip install --user --upgrade --upgrade-strategy eager pipenv wheel
  ```