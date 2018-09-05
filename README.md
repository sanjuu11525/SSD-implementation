# SSD PyTorch Implementation
The main task of this repo is to implement SSD detector with PyTorch. Actually, the reader could find out many repos having the same purpose and even providing command-line supports for the task configuration. However, from the author's perspective, parts of the implementation could not be understood easily because of inconsistent coding styles by contributors. Moreover, the flexibility of python somehow is abused, then very confusing especially in operating diemnsions of tensor. In this repo, documentation is added carefully. Currently, without augumented data the evaluation with Pascal 2007 is 65%, which is consistent with the published. More content will be added in the future.

## Dependencies
PyTorch v0.4.0
OpenCV

## Reference
[1]https://github.com/amdegroot/ssd.pytorch

[2]https://github.com/pytorch/vision#models
