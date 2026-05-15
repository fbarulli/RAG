---
id: afd037e097
question: 'Pipenv: Jupyter no output'
sort_order: 8
---

**Problem:** I tried to run a starter notebook in a Pipenv environment but had issues with no output on prints. I used `scikit-learn==1.2.2` and `python==3.10`. Tornado version was `6.3.2`.

**Solution:** The error you're encountering seems to be a bug related to Tornado, which is a Python web server and networking library. It's used by Jupyter under the hood to handle networking tasks.

- Downgrading to `tornado==6.1` fixed the issue.

More information can be found on [this Stack Overflow post](https://stackoverflow.com/questions/54971836/no-output-jupyter-notebook).