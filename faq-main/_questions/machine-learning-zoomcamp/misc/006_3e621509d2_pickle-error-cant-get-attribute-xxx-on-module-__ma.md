---
id: 3e621509d2
question: 'Pickle error: can’t get attribute XXX on module __main__'
sort_order: 6
---

When running a Docker container with Waitress serving the `app.py` for making predictions, you may encounter a pickle error: "can't get attribute `<name_of_class>` on module __main__".

This error does not occur when Flask is used directly, i.e., not through Waitress.

**Cause**

The issue arises because the model uses a custom column transformer class. When the model was saved, it was saved from the `__main__` module (e.g., via `python train.py`). Pickle references the class in the global namespace (top-level code): `__main__.<custom_class>`.

When using Waitress, it loads the `predict_app` module, and this calls `pickle.load`, which tries to find `__main__.<custom_class>`, but it does not exist in that namespace.

**Solution**

1. Move the custom class into a separate module.
2. Import this module in both the script that saves the model (e.g., `train.py`) and the script that loads the model (e.g., `predict.py`).

**Note:** If Flask is used (without Waitress) in `predict.py`, and `predict.py` defines the class, executing `python predict.py` will work because the class is in the same namespace as when the model was saved (`__main__`).

For more information, check out the [detailed explanation on Stack Overflow](https://stackoverflow.com/questions/27732354/unable-to-load-files-using-pickle-and-multiple-modules).