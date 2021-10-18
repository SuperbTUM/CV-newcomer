## Introduction

Archive of [Ali TianChi/DataWhale Computer Vision Competition](https://tianchi.aliyun.com/competition/entrance/531795/information?lang=en-us) --- The Street View House Numbers (SVHN, Google) Dataset Challenge.

Architecture: PyTorch (with cuda and cudnn, Dataloader is overrided with prefetch), OpenCV

Data Augmentation: image sharpening, rotation, translation, normalization, blur (gaussian, average, median), superpixel, emboss, spatial augmentaion (HSV), convolutional augmentation (not good), elastic transformation (often happen in MNIST dataset), elementwise add, noise, dropout --- imgaug is able to handle all of them

Backbone: GRU/BiLSTM (baseline, score: 0.643), darknet (first attempt, yolov4-tiny, anchor re-organized, mish activation, DIOU NMS, iterations for 10000 iterations, score: 0.833), pretrained ResNet-50 (Mish/Meta-ACON (CVPR 2021), LeakyRelu) with dropblock (may not helpful in FC layer)/dropout (second attempt, label smoothing, simple voting algorithm, score: 0.847, ranking: top 4%, single character recognition improved dramatically)

Optimizer: SGD? (Performs bad?, Adam is better) with Cosine learning rate scheduler

Loss Function: label smoothing loss
