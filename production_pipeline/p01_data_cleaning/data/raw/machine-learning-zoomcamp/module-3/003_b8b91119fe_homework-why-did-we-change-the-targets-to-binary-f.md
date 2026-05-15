---
id: b8b91119fe
question: 'Homework: Why did we change the targets to binary format when calculating
  mutual information score in the homework?'
sort_order: 3
---

Mutual Information score calculates the relationship between categorical or discrete variables. In the homework, the target, `median_house_value`, was continuous. Thus, we changed it to a binary format to make its values discrete (either 0 or 1).

Keeping the target as a continuous variable would require the algorithm to divide it into bins, which would be highly subjective. This is why continuous variables are not used for mutual information score calculation.