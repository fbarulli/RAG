---
id: 56ebfe8111
question: 'Homework: How do I import data from ''bank-full.csv''?'
sort_order: 1
---

Import the data using the following command:

```python
import pandas as pd

df = pd.read_csv("bank-full.csv", sep=';')
```

Note that the data is separated by a semicolon, not a comma.