Archive of aliyun coding competition -- newcomer level.

URL: https://tianchi.aliyun.com/competition/entrance/531795/information

Architecture: PyTorch (Dataloader is overrided to enhance multiprocessing performance) and OpenCV

Data Augmentation: image sharpening, rotation, translation, normalization

Detection Network: YoloV4 (pending), GRU/BiLSTM (baseline), darknet (baseline, score: 0.792, ranking: top 5%)

Classification Network: pretrained ResNet-50 (Mish/Meta-ACON (CVPR 2021), LeakyRelu) with dropblock/dropout

Optimizer: SGD with Cosine learning rate scheduler

Loss Function: label smoothing loss
