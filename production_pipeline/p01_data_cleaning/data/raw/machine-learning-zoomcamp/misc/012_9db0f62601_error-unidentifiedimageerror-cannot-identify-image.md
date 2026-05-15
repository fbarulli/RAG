---
id: 9db0f62601
question: 'Error: UnidentifiedImageError: cannot identify image file'
sort_order: 12
---

In deploying the model, I encountered an error while testing my model locally on a test-image data.

The initial command was:

```python
url = '[GitHub](https://github.com/bhasarma/kitchenware-classification-project/blob/main/test-image.jpg')

X = preprocessor.from_url(url)
```

The error received:

```python
UnidentifiedImageError: cannot identify image file <_io.BytesIO object at 0x7f797010a590>
```

**Solution:**

- Add `?raw=true` after `.jpg` in the URL. For example:

  ```python
  url = '[GitHub](https://github.com/bhasarma/kitchenware-classification-project/blob/main/test-image.jpg?raw=true)'
  ```