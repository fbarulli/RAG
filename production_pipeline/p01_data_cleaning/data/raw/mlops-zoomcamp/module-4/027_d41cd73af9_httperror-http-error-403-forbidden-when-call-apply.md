---
id: d41cd73af9
question: 'HTTPError: HTTP Error 403: Forbidden when call apply_model() in score.ipynb'
sort_order: 27
---

Solution:

Instead of using the following input file:

```python
input_file = f'https://s3.amazonaws.com/nyc-tlc/trip+data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
```

Use:

```python
input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
```