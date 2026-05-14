---
id: 3547bb9442
question: Shuffling data using Numpy’s Generator Feature
sort_order: 16
---

While the lectures have you use the shuffle function to shuffle the index of the dataframe, it no longer accepts random seed as a parameter. This is because Numpy converted this feature into its own "[Generator](https://numpy.org/doc/stable/reference/random/generator.html) Class". In order to assign the random generator a seed, you have to specify the object (`rng`) that you are going to utilize in your code:

```python
# Create index from range of values in array
idx = np.arange(n)

# Create random generator object and set seed
rng = np.random.default_rng(random_seed)

# Shuffle values using Generator object
rng.shuffle(idx)
```