---
id: ebd339f9a5
question: Error when importing evidently package because of numpy version upgraded
sort_order: 17
---

A new version of Numpy has just been released v 2.0.0 (on Jun 16, 2024), which causes an import error of the package.

```python
"`np.chararray` is deprecated and will be removed from "

419     "the main namespace in the future. Use an array with a string "

420     "or bytes dtype instead.", DeprecationWarning, stacklevel=2): `np.float_` was removed in the NumPy 2.0 release. Use `np.float64` instead.
```

Or

```python
AttributeError: `np.float_` was removed in the NumPy 2.0 release. Use `np.float64` instead.
```

You can solve it by reinstalling numpy to a previous version 1.26.4. Just run:

```bash
python -m pip install numpy==1.26.4
```

Or modify the `requirements.txt` to freeze the version:

```text
numpy==1.26.4
```