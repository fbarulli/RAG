---
id: 85c83560c5
question: Creating Helper functions in Mage
sort_order: 11
---

There’s no need to add the utility functions in each sub-project when you watch the videos as there only needs to be one set. Just verify the code is still the same as in Mage’s mlops repository.

As for the import statements:

```python
from mlops.utils.[...] import [...]
```

All refer to the same path in the main mlops "parent" project:

```
/[mage-mlops-repository-name]/mlops/utils/...
```