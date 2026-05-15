---
id: 8a5d89cbae
question: Saving and loading Xgboost
sort_order: 24
---

If you have problems with pickling the models, you 
can use an alternative approach.

Save model by calling `save_model`, load with `load_model`:

```python
model.save_model('model.bin')

...

bst = xgb.Booster()
bst.load_model('model.bin')
```