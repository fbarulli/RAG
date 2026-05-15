---
id: b7e73fcee7
question: Floating Point Precision
sort_order: 1
---

I was doing Question 7 from Week 1 Homework and with step 6: Invert XTX. I created the inverse. Now, an inverse when multiplied by the original matrix should return an identity matrix. But when I multiplied the inverse with the original matrix, it gave a matrix like this:

Inverse * Original:

```
[[ 1.00000000e+00 -1.38777878e-16]
 [ 3.16968674e-13  1.00000000e+00]]
```

Solution:

It's because floating point math doesn't work well on computers as shown here: [https://stackoverflow.com/questions/588004/is-floating-point-math-broken](https://stackoverflow.com/questions/588004/is-floating-point-math-broken)
