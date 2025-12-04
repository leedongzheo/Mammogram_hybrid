# hybrid-ds
Implementation of the paper "A Unified Mammogram Analysis Method via Hybrid Deep Supervision".

This implementation now targets PyTorch v2.8.0.

Suggested core dependencies:

* `torch==2.8.0`
* `torchvision==0.19.0`
* `imageio`

Label expectations:

* Place segmentation masks under `data/label` with the same base filename as the image.
* Masks are treated as binary: background pixels are `0` and lesion pixels are any value > 0.
* For a normal (negative) image with no mass, the mask should therefore be entirely zeros.

Before running the code by conducting `python train.py`, please store your images and pixel-wise labels in data/img and data/label respectively, and modify the values in data/meanstd.txt according to your data.
