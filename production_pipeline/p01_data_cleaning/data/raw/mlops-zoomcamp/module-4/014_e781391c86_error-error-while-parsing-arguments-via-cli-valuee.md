---
id: e781391c86
question: 'Error: Error while parsing arguments via CLI [ValueError: Unknown format
  code ''d'' for object of type ''str'']'
sort_order: 14
---

When passing arguments to a script via command line and converting it to a 4-digit number using `f’{year:04d}’`, this error can occur.

This happens because command line inputs are read as strings. They need to be converted to integers before formatting with an f-string:

```python
year = int(sys.argv[1])
f'{year:04d}'
```

If you use the `click` library, update your decorator accordingly:

```python
import click

@click.command()
@click.option("--year", help="Year for evaluation", type=int)
def your_function(year):
    # Your code
```