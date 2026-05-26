---
id: d2cdec8f35
question: Convert dictionary values to Dataframe table
sort_order: 17
---

You can convert the prediction output values to a DataFrame using the following code:

```python
import pandas as pd

df = pd.DataFrame.from_dict(your_dict, orient='index', columns=['Prediction'])
```