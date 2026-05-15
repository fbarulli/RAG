---
id: b7625d600a
question: 'psycopg.OperationalError: connection failed: connection to server at "127.0.0.1",
  port 5432 failed: FATAL:  password authentication failed for user "postgres"'
sort_order: 27
---

It could be that there is already another Docker container running (for example, from a previous session).

To resolve this issue:

1. Check for running containers:
   ```bash
   docker ps
   ```
2. Stop the running container:
   ```bash
   docker stop <container_name_or_ID>
   ```