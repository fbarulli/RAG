---
id: 16099520dd
question: 'AWS: Regions need to match in docker-compose'
sort_order: 16
---

**Problem Description**

If you are experiencing issues with integration tests and Kinesis, ensure that your AWS regions are consistent between docker-compose and your local configuration. Otherwise, you may create a stream in an incorrect region.

**Solution Description**

- Set the region in your AWS config file:
  
  ```plaintext
  ~/.aws/config
  ```
  
  Example:
  
  ```text
  region = us-east-1
  ```

- Ensure that the region in your `docker-compose.yaml` is also set:
  
  ```yaml
  environment:
    - AWS_DEFAULT_REGION=us-east-1
  ```