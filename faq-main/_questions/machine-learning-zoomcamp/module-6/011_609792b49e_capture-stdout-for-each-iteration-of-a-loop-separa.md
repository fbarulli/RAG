---
id: 609792b49e
question: Capture stdout for each iteration of a loop separately
sort_order: 11
---

I wanted to directly capture the output from the XGBoost training for multiple eta values to a dictionary without needing to run the same cell multiple times, edit the eta value manually, or copy the code for different eta values.

Using the magic cell command `%%capture output`, I could only capture the complete output for all iterations of the loop. I was able to solve this using the following approach. This is just a code sample to illustrate the idea:

```python
from IPython.utils.capture import capture_output
import sys

different_outputs = {}

for i in range(3):
    with capture_output(sys.stdout) as output:
        print(i)
        print("testing capture")
    different_outputs[i] = output.stdout

# Output:
# different_outputs
# {0: '0\ntesting capture\n',
#  1: '1\ntesting capture\n',
#  2: '2\ntesting capture\n'}
```
