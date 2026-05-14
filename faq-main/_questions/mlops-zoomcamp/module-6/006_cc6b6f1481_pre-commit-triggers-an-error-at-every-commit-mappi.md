---
id: cc6b6f1481
question: 'Pre-commit: Triggers an error at every commit: “mapping values are not
  allowed in this context”'
sort_order: 6
---

At every commit, the above error is thrown and no pre-commit hooks are run.

Ensure the indentation in `.pre-commit-config.yaml` is correct, particularly the 4 spaces ahead of every `repo` statement.