---
id: b53176b742
question: 'Error: pipenv command not found after pipenv installation'
sort_order: 29
---

When installing pipenv using the `--user` option, you need to update the PATH environment variable to run pipenv commands. It's recommended to update your `.bashrc` or `.profile` (depending on your OS) to persist the change. Edit your `.bashrc` file to include or update a line like this:

```bash
PATH="<path_to_your_pipenv_install_dir>:$PATH"
```

Alternatively, you can reinstall pipenv as root for all users:

```bash
sudo -H pip install -U pipenv
```