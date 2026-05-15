---
id: ba50d9e0bc
images:
- description: 'image #1'
  id: image_1
  path: images/mlops-zoomcamp/image_f76f8e53.png
question: Mage in Codespaces in a subfolder under mlops-zoomcamp repository
sort_order: 5
---

**Issue 1:** Errors such as:

```bash
[+] Running 1/1

✘ magic-database Error too many requests: You have reached your pull rate limit. You may increase the limit by authenticating and upgra...                   

Error response from daemon: too many requests: You have reached your pull rate limit. You may increase the limit by authenticating and upgrading: [docker.com](https://www.docker.com/increase-rate-limit)
```

**Issue 2:** Popups with different percentage values indicating space is in single digits.

<{IMAGE:image_1}>

**Solution:** It is not recommended to set up Mage as a subfolder of mlops-zoomcamp. See findings in this thread for more information.