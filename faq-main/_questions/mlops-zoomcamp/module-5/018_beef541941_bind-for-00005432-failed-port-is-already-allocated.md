---
id: beef541941
question: 'Bind for 0.0.0.0:5432 failed: port is already allocated'
sort_order: 18
---

**Problem:** When trying to start the postgres services through `docker-compose up`, this error occurs:

```
Bind for 0.0.0.0:5432 failed: port is already allocated
```

**Note:** This issue occurs because port 5432 is already used by another service.

**Solution:** Update the port mapping for the Postgres service to `5433:5432` in the Docker Compose YAML file.