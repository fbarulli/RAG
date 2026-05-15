---
id: 26a7d0d9b6
question: Adminer Not Loaded
sort_order: 20
---

**Problem:** After running Docker Compose, Adminer cannot be accessed on [http://127.0.0.1:8080/](http://127.0.0.1:8080/index.php)

**Solution:** Add `index.php` after the URL, so the URL will be [http://127.0.0.1:8080/index.php](http://127.0.0.1:8080/index.php)