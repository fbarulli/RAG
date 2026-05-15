---
id: b72a4be56d
question: 'How to avoid Value errors with array shapes in homework?'
sort_order: 2
---

First of all, use `np.dot` for matrix multiplication. When you perform matrix-matrix multiplication, remember that the order of multiplication is crucial and affects the result.

**Dimension Mismatch**

To perform matrix multiplication, the number of columns in the first matrix should match the number of rows in the second matrix. Rearrange the order to satisfy this condition.