---
id: cf068009f1
question: 'AWS EC2: How do I handle changing IP addresses on restart?'
sort_order: 16
---

Every time I restart my EC2 instance, I receive a different IP and need to update the config file manually.

**Solution:**

You can create a script to automatically update the IP address of your EC2 instance. Refer to this [guide](https://github.com/dimzachar/mlops-zoomcamp/blob/master/notes/Week_1/update_ssh_config.md) for detailed steps.