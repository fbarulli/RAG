---
id: dda62d0ef0
question: 'Storage: Bucket Permission Denied Error when running the gcp_setup flow'
sort_order: 9
---

When following the [YouTube lesson](https://www.youtube.com/watch?v=nKqjjLJ7YXs&list=PL3MmuxUbc_hJed7dXYoJw8DoCuVHhGEQb&index=23) and then running the [gcp_setup flow](https://github.com/DataTalksClub/data-engineering-zoomcamp/blob/main/02-workflow-orchestration/flows/05_gcp_setup.yaml), you might encounter a permission denied error.

To resolve this:

1. Verify if the bucket already exists using the GCP console.
2. If it exists, choose a different name for the bucket.

**Note:** The GCP bucket name must be unique globally across all buckets, as the bucket will be accessible by URL.
