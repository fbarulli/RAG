---
id: cdd6cd3f29
question: 'Homework: Parameters Mismatch in Homework Q3'
sort_order: 20
---

I was using an old version of sklearn, which caused a mismatch in the number of parameters. In the latest version, `min_impurity_split` for `RandomForestRegressor` was deprecated. Upgrading to the latest version resolved the issue.