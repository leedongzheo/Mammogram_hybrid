# hybrid-ds
Implementation of the paper "A Unified Mammogram Analysis Method via Hybrid Deep Supervision".

This implementation now targets PyTorch v2.8.0.

Suggested core dependencies:

* `torch==2.8.0`
* `torchvision==0.19.0`
* `imageio`

Before running the code by conducting `python train.py`, please store your images and pixel-wise labels in data/img and data/label respectively, and modify the values in data/meanstd.txt according to your data.
