---
id: 007cc84471
question: 'Failed to listen on :::8080 (reason: php_network_getaddresses: getaddrinfo
  failed: Address family for hostname not supported)'
sort_order: 15
---

Problem: When running `docker-compose up --build`, you may encounter this error.

To solve this issue, add the following command in the `adminer` block in your `docker-compose.yml` file:

```yaml
a dminer:
  command: php -S 0.0.0.0:8080 -t /var/www/html
  image: adminer...
```

This configuration specifies the command to be executed when the container starts, setting up PHP to listen on `0.0.0.0:8080`. This addresses the network error by changing the bind address.