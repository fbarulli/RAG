---
id: 9fbb36bb16
question: Alternative way to load the data using requests
sort_order: 8
---

Here's another way to load a dataset using the `requests` library:

```python
import requests

url = 'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv'

response = requests.get(url)

if response.status_code == 200:
    with open('housing.csv', 'wb') as file:
        file.write(response.content)
else:
    print("Download failed.")
```