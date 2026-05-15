---
id: 25ff7dd14c
question: 'Error decoding JSON response: Expecting value: line 1 column 1 (char 0)'
sort_order: 14
---

This problem occurs when contacting the server to send your predict-test data in the correct shape. The issue is that the input format to the model wasn't in the right shape.

The server receives data in JSON format (dict), which is not suitable for the model. You should convert it to a format like numpy arrays.