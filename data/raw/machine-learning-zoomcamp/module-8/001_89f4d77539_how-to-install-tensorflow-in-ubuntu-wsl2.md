---
id: 89f4d77539
question: How to install Tensorflow in Ubuntu WSL2
sort_order: 1
---

Running a CNN on your CPU can take a long time, and once you’ve run out of free time on some cloud providers, it’s time to pay up. Both can be tackled by installing TensorFlow with CUDA support on your local machine if you have the right hardware.

I was able to get it working by using the following resources:

- [CUDA on WSL :: CUDA Toolkit Documentation (nvidia.com)](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [Install TensorFlow with pip](https://www.tensorflow.org/install/pip#windows-wsl2)
- [Start Locally | PyTorch](https://pytorch.org/get-started/locally/)

I included the link to PyTorch so that you can get that one installed and working too while everything is fresh on your mind. Just select your options, and for Computer Platform, I chose CUDA 11.7 and it worked for me.

If you need GPU support, check this article: [https://knowmledge.com/2023/12/07/ml-zoomcamp-2023-project/](https://knowmledge.com/2023/12/07/ml-zoomcamp-2023-project/)