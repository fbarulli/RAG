---
id: a0ac7f4f77
question: 'Error importing xgboost in python with OS mac: library not loaded: @rpath/libomp.dylib'
sort_order: 49
---

To fix this error, run the following command:

```bash
brew install libomp
```