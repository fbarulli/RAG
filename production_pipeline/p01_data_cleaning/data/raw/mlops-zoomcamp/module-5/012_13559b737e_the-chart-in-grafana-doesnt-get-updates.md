---
id: 13559b737e
question: The chart in Grafana doesn’t get updates
sort_order: 12
---

**Problem Description:** While my metric generation script was still running, I noticed that the charts in Grafana don’t get updated.

**Solution:**

- **Refresh Interval:** Set it to a small value, such as 5, 10, or 30 seconds.
- **Timezone Setting:** Ensure you use your local timezone in a call to `pytz.timezone`. For example, change the setting from "Europe/London" to your local timezone to get updates.