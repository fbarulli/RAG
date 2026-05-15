---
id: 3fd406c55c
images:
- description: 'image #1'
  id: image_1
  path: images/mlops-zoomcamp/image_e822dac8.png
question: 'SSH: Connection to AWS EC2 instance from local machine WSL getting terminated
  frequently within a minute of inactivity.'
sort_order: 22
---

If the SSH connection from your local machine’s WSL to an AWS EC2 instance is frequently getting terminated after a short period of inactivity, you might see the following message displayed:

<{IMAGE:image_1}>

To fix this issue, add the following lines to your `config` file in the `.ssh` directory of your WSL environment:

```
ServerAliveInterval 60
ServerAliveCountMax 3
```

For example, after adding these lines, your SSH configuration should look somewhat like this:

```
Host mlops-zoomcamp
  HostName 45.80.32.7
  User ubuntu
  IdentityFile ~/.ssh/siddMLOps.pem
  StrictHostKeyChecking no
  ServerAliveInterval 60
  ServerAliveCountMax 3
```