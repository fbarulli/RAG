---
id: 2ce73ee58a
question: Get an error ‘Unable to locate credentials’ after running localstack with
  kinesis
sort_order: 3
---

You may encounter the error `{'errorMessage': 'Unable to locate credentials', …` from the print statement in `test_docker.py` after running localstack with Kinesis.

To resolve this issue:

1. In the `docker-compose.yaml` file, add the following environment variables:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   
   You can assign any value to these variables (e.g., `abc`).

2. Alternatively, you can run the following command:

   ```bash
   aws --endpoint-url http://localhost:4566 configure
   ```

   Provide random values for the following prompts:
   - AWS Access Key ID
   - AWS Secret Access Key
   - Default region name
   - Default output format

<END>