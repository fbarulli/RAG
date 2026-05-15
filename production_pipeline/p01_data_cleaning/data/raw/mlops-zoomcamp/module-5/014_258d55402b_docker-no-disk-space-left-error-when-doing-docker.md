---
id: 258d55402b
question: 'Docker: no disk space left error when doing docker compose up'
sort_order: 14
---

To resolve the "no disk space left" error when running `docker compose up`, follow these steps:

1. Run the following command to remove unused objects (build cache, containers, images, etc.):
   
   ```bash
   docker system prune
   ```

2. If you want to see what is taking up space before pruning, use:
   
   ```bash
   docker system df
   ```