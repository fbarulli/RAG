---
id: 5c68f358c2
question: Generate Evidently Chart in Grafana
sort_order: 16
---

**Problem:** Can we generate charts like Evidently inside Grafana?

**Solution:**

- In Grafana, you can use a stat panel (just a number) and a scatter plot panel, which may require a plug-in.
- Unfortunately, there's no native method to directly recreate the Evidently dashboard.
- Ensure that all relevant information is logged to your Grafana data source, then design your custom plots.

**External Recreation:**

- Export the Evidently output in JSON with `include_render=True` for external visualization:
  - See more details [here](https://docs.evidentlyai.com/user-guide/customization/json-dict-output).
- For non-aggregated visuals, use the option "`raw_data": True".
  - More details [here](https://docs.evidentlyai.com/user-guide/customization/report-data-aggregation).

This specific plot with under- and over-performance segments is particularly useful during debugging and might be easier to view ad hoc using Evidently.