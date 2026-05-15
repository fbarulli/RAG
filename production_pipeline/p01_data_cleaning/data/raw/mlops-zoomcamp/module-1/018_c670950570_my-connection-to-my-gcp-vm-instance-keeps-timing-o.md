---
id: c670950570
question: My connection to my GCP VM instance keeps timing out when I try to connect
sort_order: 18
---

If you switched off the VM instance completely in GCP, the IP address may change when it switches back on. You need to update the `ssh_config` file with the new external IP address. This can be done in VS Code if you have the Remote-SSH extension installed.

1. Open the command palette and type `Remote-SSH: Open SSH Configuration File…`.
2. Select the appropriate `ssh_config` file.
3. Edit the `HostName` to the correct IP address.