# SSD PyTorch Implementation
The main task of this repo is to reimplement SSD detector with PyTorch. Actually, the reader could find out many having the same purpose and even providing command-line supports for the task configuration. However, from the author's perspective, parts of the implementation could not be understood easily because of inconsistent coding styles by contributors. Moreover, the flexibility of python is sometime abused. The author has been very confused when dealing with redundant diemnsion manipulation of tensor. In this repo, the straightforward implementation with documentation is done carefully. Sharing the experience is the main purpose. If something incorrect, please send me messages by email.

Regarding the evaluation, without augmented data the evaluation with Pascal 2007 is 65%, which is consistent with the published. More content will be added in the future.

## Dependencies
PyTorch v0.4.0

OpenCV for image io
## Training set
Pascal 2007+2012 trainval

## Pascal 2007 eval
The training consists of two parts. Because of lacking hardware resource, the pretraining of VGG model is used. Respect to [1], the reader can download the model and make a try. The second part is to train the extra layers and tun weights. The evaluation without augmented is below.

|mAP|aeroplane|bicycle|bird|boat|bottle|bus|car|cat|chair|cow|diningtable|dog|horse|motorbike|person|pottedplant|sheep|sofa|train|tvmonitor|
|--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|0.658|0.721|0.726|0.614|0.579|0.358|0.714|0.805|0.798|0.495|0.635|0.617|0.734|0.758|0.737|0.786 |0.446|0.599|0.626|0.776|0.634|

## Reference
[1]https://github.com/amdegroot/ssd.pytorch

[2]https://github.com/kuangliu/torchcv
