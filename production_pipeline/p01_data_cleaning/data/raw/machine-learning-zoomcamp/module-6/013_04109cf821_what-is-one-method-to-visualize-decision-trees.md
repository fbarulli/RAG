---
id: 04109cf821
question: What is one method to visualize decision trees?
sort_order: 13
---

To visualize decision trees, you can use Graphviz along with Scikit-learn's `export_graphviz` method and `plot_tree` function.

Here are two approaches:

1. **Using `export_graphviz`:**
   ```python
   from sklearn import tree
   import graphviz

   dot_data = tree.export_graphviz(regr, out_file=None,
                                   feature_names=boston.feature_names,
                                   filled=True)
   graphviz.Source(dot_data, format="png")
   ```

2. **Using `plot_tree`:**
   ```python
   from sklearn import tree
   
   tree.plot_tree(dt, feature_names=dv.feature_names_)
   ```

Both methods help in generating a visual representation of the decision tree.