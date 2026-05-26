---
id: e8df9f0d12
question: 'Docker: When trying to run a streamlit app using docker-compose, I get:
  Error response from daemon: failed to create task for container: failed to create
  shim task: OCI runtime create failed: runc create failed: unable to start container
  process: exec: "streamlit": executable file not found in $PATH: unknown. The app
  runs fine outside of docker-compose'
sort_order: 1
---

To resolve this issue:

1. Ensure you have created a `Dockerfile`.
2. Add `streamlit` to the `docker-compose` configuration.
3. Run the following command to rebuild and start the application:

   ```bash
   docker-compose up --build
   ```