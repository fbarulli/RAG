---
id: d00a9ac4ba
question: 'Homework/Question 6: Entry point for running scoring script in Docker container'
sort_order: 36
---

For question 6, if you are using the script as instructed in the homework and not Flask, your entry point should be `bash`. This can be set by specifying:

```dockerfile
ENTRYPOINT = ["bash"]
```