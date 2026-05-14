---
id: 8cb6b3176b
question: Error in starting monitoring services in Linux
sort_order: 4
---

**Problem Description:**

In Linux, when starting services using `docker compose up --build` as shown in video 5.2, the services won’t start and instead we get the message:

```
unknown flag: --build
```

**Solution:**

Since we install docker-compose separately in Linux, use the following command:

```bash
docker-compose up --build
```