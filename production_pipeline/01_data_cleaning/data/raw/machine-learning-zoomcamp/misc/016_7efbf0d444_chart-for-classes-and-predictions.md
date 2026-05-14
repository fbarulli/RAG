---
id: 7efbf0d444
question: Chart for classes and predictions
sort_order: 16
---

How to visualize the predictions per classes after training a neural net:

```python
classes, predictions = zip(*dict(zip(classes, predictions)).items())

plt.figure(figsize=(12, 3))

plt.bar(classes, predictions)
```