---
id: fa23739c18
question: Empty Records in Kinesis Get Records with LocalStack
sort_order: 8
---

**Problem Description**

Following video 6.3, at minute 11:23, the get records command returns empty records.

**Solution**

Add `--no-sign-request` to the Kinesis get records call:

```bash
aws --endpoint-url=http://localhost:4566 kinesis get-records --shard-iterator [...] --no-sign-request
```