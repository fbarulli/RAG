---
id: 4d08d35199
question: 'AttributeError: ''MlflowClient'' object has no attribute ''list_run_infos'''
sort_order: 46
---

Problem: In the scenario 2 notebook, the error 

```python
AttributeError: 'MlflowClient' object has no attribute 'list_run_infos'
```

is thrown when running:

```python
run_id = client.list_run_infos(experiment_id='1')[0].run_id
```

Solution: Use the following code instead:

```python
run_id = client.search_runs(experiment_ids='1')[0].info.run_id
```

Scenario: This solution works for MLflow version 2.12.2 and might work for other recent versions as of May, 2024.