---
id: 93dfb7d96c
images:
- description: 'image #1'
  id: image_1
  path: images/mlops-zoomcamp/image_03216ab8.png
question: 'Grafana dashboard error after reset: db query error: pq: database “test”
  does not exist'
sort_order: 23
---

Problem: You’ve already loaded your data, created a dashboard, and saved it. However, upon running `docker-compose up` after saving the dashboard, you encounter this error: 

```plaintext
db query error: pq: database “test” does not exist
```

<{IMAGE:image_1}>

Solution:

This error indicates you haven’t run the DB initialization code. If you did run it before and even saw results, the issue likely arises because you restarted the docker-compose services.

The default `docker-compose.yml` file doesn’t have a volume for the Postgres DB. This means every restart will delete the DB data.

To resolve this:

1. **If not planning to restart the services again:** Simply rerun the DB initialization and filling code of your exercise.

2. **If you plan to restart services frequently:**
   - Add a volume to your PostgreSQL service in the `docker-compose.yml` file:

     ```yaml
     volumes:
       - ./data/postgres:/var/lib/postgresql/data
     ```

   - Note: Ensure you create a `./data` directory in your project.

3. To attach the volume, run the following:

   ```bash
   docker-compose down
   docker-compose up --build
   ```