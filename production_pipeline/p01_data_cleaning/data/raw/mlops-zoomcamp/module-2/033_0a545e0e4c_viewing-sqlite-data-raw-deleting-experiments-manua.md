---
id: 0a545e0e4c
images:
- description: 'image #1'
  id: image_1
  path: images/mlops-zoomcamp/image_10815a9d.png
- description: 'image #2'
  id: image_2
  path: images/mlops-zoomcamp/image_e3defff0.png
question: Viewing SQLite Data Raw & Deleting Experiments Manually
sort_order: 33
---

All the experiment and other tracking information in MLflow are stored in an SQLite database provided while initiating the `mlflow ui` command. This database can be inspected using PyCharm’s Database tab by selecting the SQLite database type.

Once the connection is created, the tables can be queried and inspected using standard SQL. The same applies to any SQL-backed database such as PostgreSQL.

This approach is useful to understand the entity structure of the data being stored within MLflow and is beneficial for systematic archiving of model tracking for extended periods.

<{IMAGE:image_1}>

<{IMAGE:image_2}>
