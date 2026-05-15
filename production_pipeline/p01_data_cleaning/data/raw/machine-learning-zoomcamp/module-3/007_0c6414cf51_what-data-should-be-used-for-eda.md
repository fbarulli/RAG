---
id: 0c6414cf51
question: What data should be used for EDA?
sort_order: 7
---

It's indeed good practice to only rely on the train dataset for EDA. Including validation might be okay. But we aren't supposed to touch the test dataset; even just looking at it isn't a good idea. We indeed pretend that this is the future unseen data.