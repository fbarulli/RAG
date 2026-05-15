---
id: c18119ba32
question: Invalid dataset ID Error when running the gcp_setup flow
sort_order: 10
---

When following the [YouTube lesson](https://www.youtube.com/watch?v=nKqjjLJ7YXs&list=PL3MmuxUbc_hJed7dXYoJw8DoCuVHhGEQb&index=23) and then running the [gcp_setup flow](https://github.com/DataTalksClub/data-engineering-zoomcamp/blob/main/02-workflow-orchestration/flows/05_gcp_setup.yaml), the error occurs during the `create_bq_dataset` task.

The error is less clear, but it stems from using a dash in the dataset name. To resolve this, change the dataset name to something like "de_zoomcamp" to avoid using a dash. This should resolve the error.