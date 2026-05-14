---
id: 0be9e7cc3d
question: Singular Matrix Error
sort_order: 5
---

I'm trying to invert the matrix but I got an error that the matrix is singular.

The singular matrix error is caused because not every matrix can be inverted. In particular, it happens when dealing with multiplication (using the method `.dot`) since multiplication is not commutative. `X.dot(Y)` is not necessarily equal to `Y.dot(X)`. Respect the order; otherwise, you get the wrong matrix.