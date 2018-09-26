# SSD PyTorch Implementation
The main task of this repo is to reimplement main components of single-shot detector. Actually, the reader could find out many having the same purpose and even more providing command-line supports for the task configuration. However, from the author's perspective, parts of the implementation could not be understood easily because of inconsistent coding styles by contributors. Moreover, the flexibility of python is sometime abused. The author has been very confused when dealing with redundant diemnsion manipulation of tensor. In this repo, the straightforward and clear implementation with documentation is done carefully. If something incorrect, please send me messages by email.

Regarding the evaluation, without augmented data the evaluation with Pascal 2007 is 65%, which is consistent with the published. More content will be added in the future.

## Dependencies
PyTorch v0.4.0 

OpenCV for image io

Jupytor Notebook for demo

## Demonstration
demo.ipynb provides a slim demonstration for training and evaluation tasks. 

## Training set
Pascal 2007+2012 trainval

## Pascal 2007 eval
The training consists of two parts. Because of lacking hardware resource, the pretraining of VGG model is used. Respect to [1], the reader can download the model and make a try. The second part is to train the extra layers and tun weights. The evaluation without augmented is below.

|mAP|aeroplane|bicycle|bird|boat|bottle|bus|car|cat|chair|cow|diningtable|dog|horse|motorbike|person|pottedplant|sheep|sofa|train|tvmonitor|
|--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|0.658|0.721|0.726|0.614|0.579|0.358|0.714|0.805|0.798|0.495|0.635|0.617|0.734|0.758|0.737|0.786 |0.446|0.599|0.626|0.776|0.634|

mAP:65%

The evaluation with random crop+flip is below

|mAP|aeroplane|bicycle|bird|boat|bottle|bus|car|cat|chair|cow|diningtable|dog|horse|motorbike|person|pottedplant|sheep|sofa|train|tvmonitor|
|--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|0.793|0.819|0.761|0.663|0.532|0.807|0.844|0.861|0.798|0.544|0.725|0.688|0.817|0.817|0.811|0.821 |0.575|0.72|0.693|0.842|0.712|

mAP:74.3%

## Todo
- [x] Implement SSD
- [x] More augmentation
 
## Reference
[1]https://github.com/amdegroot/ssd.pytorch

[2]https://github.com/kuangliu/torchcv
