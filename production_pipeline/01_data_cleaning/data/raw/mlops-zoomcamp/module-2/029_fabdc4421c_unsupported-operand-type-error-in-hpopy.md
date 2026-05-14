---
id: fabdc4421c
question: Unsupported Operand Type Error in hpo.py
sort_order: 29
---

Running the command:

```bash
python hpo.py --data_path=./your-path --max_evals=50
```

leads to the following error:

```python
TypeError: unsupported operand type(s) for -: 'str' and 'int'
```

**Full Traceback:**

```plaintext
File "~/repos/mlops/02-experiment-tracking/homework/hpo.py", line 73, in <module>
  run(args.data_path, args.max_evals)

File "~/repos/mlops/02-experiment-tracking/homework/hpo.py", line 47, in run
  fmin(

File "~/Library/Caches/pypoetry/virtualenvs/mlflow-intro-SyTqwt0D-py3.9/lib/python3.9/site-packages/hyperopt/fmin.py", line 540, in fmin
  return trials.fmin(

File "~/Library/Caches/pypoetry/virtualenvs/mlflow-intro-SyTqwt0D-py3.9/lib/python3.9/site-packages/hyperopt/base.py", line 671, in fmin
  return fmin(

File "~/Library/Caches/pypoetry/virtualenvs/mlflow-intro-SyTqwt0D-py3.9/lib/python3.9/site-packages/hyperopt/fmin.py", line 586, in fmin
  rval.exhaust()

File "~/Library/Caches/pypoetry/virtualenvs/mlflow-intro-SyTqwt0D-py3.9/lib/python3.9/site-packages/hyperopt/fmin.py", line 364, in exhaust
  self.run(self.max_evals - n_done, block_until_done=self.asynchronous)
```

**Solution:**

The `--max_evals` argument in `hpo.py` is not defined with a datatype, leading to it being interpreted as a string. It should be an integer to ensure the script functions correctly. Modify the argument definition as follows:

```python
parser.add_argument(
    "--max_evals",
    type=int,
    default=50,
    help="the number of parameter evaluations for the optimizer to explore."
)
```