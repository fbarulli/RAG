---
id: d317ebf313
question: 'Mage: Pipeline breaks with `[Errno 2] No such file or directory: ''/home/src/mage_data/{…}
  /.variables/{...}/output_1/object.joblib''`'
sort_order: 3
---

1. Export the pipeline as a zip file.
2. Create a new Mage project.
3. Import the pipeline zip to the new project.

Additionally, check the following:

- Review the logs of the upstream block that was expected to generate `object.joblib`. Ensure it completed successfully.
- Verify that the expected output (often named `output_1`) was created and saved.
- Check in the Mage UI or directly in the file system (if accessible) to confirm whether the file exists in the `.variables` directory for that upstream block.