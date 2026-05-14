---
id: 60da2ce9c6
question: 'Docker: Update docker-compose to initiate Mage'
sort_order: 4
---

When running `./scripts/start.sh`, the following error occurs:

```plaintext
ERROR: The Compose file './docker-compose.yml' is invalid because:

Unsupported config option for networks: 'app-network'

Unsupported config option for services: 'magic-platform'
```

Solution:

1. **Download the latest version of Docker Compose**

   ```bash
   sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   ```

2. **Apply executable permissions to the binary**

   ```bash
   sudo chmod +x /usr/local/bin/docker-compose
   ```