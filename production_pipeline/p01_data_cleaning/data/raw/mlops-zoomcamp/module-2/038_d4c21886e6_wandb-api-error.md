---
id: d4c21886e6
question: WandB API error
sort_order: 38
---

**Problem:** When running the `preprocess_data.py` file, you encounter the following error:

```bash
wandb: ERROR api_key not configured (no-tty). call wandb.login(key=[your_api_key])
```

**Solution:**

1. Go to your WandB profile and navigate to user settings.
2. Scroll down to the “Danger Zone” and copy your API key.
3. Before running `preprocess_data.py`, add and run the following cell in your notebook:

   ```bash
   %%bash
   wandb login <YOUR_API_KEY_HERE>
   ```