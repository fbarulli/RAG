---
id: '1913720953'
question: 'Downloading the data from the NY Taxis datasets gives error: 403 Forbidden'
sort_order: 38
---

Problem: While following the steps in the videos, you may encounter a 403 Forbidden error when trying to download files using `wget`.

Solution:

- The issue occurs because the links point to files on cloudfront.net. An example of such a link is:
  ```
  https://d37ci6vzurychx.cloudfront.net/trip+data/green_tripdata_2021-01.parquet
  ```

- Instead of downloading the dataset directly from the link, use the dataset URL in your file.

- Update (27-May-2023):
  - You can now download the data from the official NYC trip record page: [TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
  - Go to the page, right-click, and use "copy link" to get the URL since the URL provided might change if NYC updates their system.

- Example command:
  ```bash
  wget https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2021-01.parquet
  ```