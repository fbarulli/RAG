---
id: 3ef50b0d88
question: 'Homework Q7: Multiplication operators.'
sort_order: 7
---

Matrix multiplication, such as matrix-matrix or matrix-vector multiplication, is often represented using the `*` operator. However, in NumPy, it is performed using the `@` operator or `np.matmul()`. The `*` operator in NumPy is used for element-wise multiplication, also known as the [Hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)).

For matrix-matrix multiplication, using the `@` operator or `np.matmul()` is preferred, as noted in the [NumPy documentation](https://numpy.org/doc/stable/reference/generated/numpy.dot.html#numpy.dot).

When multiplying by a scalar, it's preferred to use `numpy.multiply()` or the `*` operator.

References:
- [numpy.dot()](https://numpy.org/doc/stable/reference/generated/numpy.dot.html)
- [ndarray.dot()](https://numpy.org/doc/1.21/reference/generated/numpy.ndarray.dot.html)