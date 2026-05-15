---
id: 5ad72730d6
images:
- description: 'image #1'
  id: image_1
  path: images/mlops-zoomcamp/image_a92c0f4d.png
question: Table/database not showing on Grafana dashboard
sort_order: 19
---

**Problem:**

For version 5.4, when trying to create a new dashboard, Grafana does not list the `dummy_metrics` table in the query tab.

**Note:** Change the datasource name from the default "PostgreSQL."

**Solution 1:**

Update the `config/grafana_datasources.yaml` with the following:

```yaml
# List of datasources to insert/update
# Available in the database
datasources:

- name: NewPostgreSQL
  type: postgres
  url: db:5432
  user: postgres
  secureJsonData:
    password: 'example'
  jsonData:
    sslmode: 'disable'
    database: test
```

**Solution 2:**

- Use the "Code" option rather than the "Builder" option.
- Load the data using your own SQL queries. See the screenshot below (box highlighted in red).
- Tip: If you write your `FROM` statement first, the `SELECT` options can be done through auto-complete.

<{IMAGE:image_1}>