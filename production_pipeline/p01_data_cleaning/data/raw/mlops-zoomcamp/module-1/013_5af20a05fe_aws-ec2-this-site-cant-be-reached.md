---
id: 5af20a05fe
question: 'AWS EC2: this site can’t be reached'
sort_order: 13
---

When I click an open IP address in an AWS EC2 instance, I get an error: "This site can’t be reached." What should I do?

This IP address is not meant to be opened in a browser. It is used to connect to the running EC2 instance via terminal. Use the following command from your local machine or a remote server:

- Assume the IP address is `11.111.11.111`
- The downloaded key name is `razer.pem` (ensure the key is moved to a hidden folder `.ssh`)
- Your username is `user_name`

```bash
ssh -i /Users/user_name/.ssh/razer.pem ubuntu@11.111.11.111
```