---
id: 17eeab7562
question: Loading the Image with PILLOW library and converting to numpy array
sort_order: 2
---

To load an image using the PILLOW library and convert it to a NumPy array, you can follow these steps:

1. **Install the Pillow library:**
   
   ```bash
   pip install pillow
   ```

2. **Use the following code to load an image and convert it:**

   ```python
   from PIL import Image
   from numpy import asarray

   # Open the image file
   img = Image.open('aeroplane.png')

   # Convert the image to a NumPy array
   numdata = asarray(img)
   ```