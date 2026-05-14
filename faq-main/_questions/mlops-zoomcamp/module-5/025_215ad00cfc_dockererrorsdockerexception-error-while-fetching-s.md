---
id: 215ad00cfc
question: 'docker.errors.DockerException: Error while fetching server API version:
  HTTPConnection.request() got an unexpected keyword argument ''chunked'''
sort_order: 25
---

- Instead of using:
  
  ```bash
  docker-compose up --build
  ```
  
- Use:

  ```bash
  docker compose up --build
  ```