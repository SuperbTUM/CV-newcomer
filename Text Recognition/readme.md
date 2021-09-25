Archive of aliyun coding competition -- newcomer level.

URL: https://tianchi.aliyun.com/competition/entrance/531795/information

Architecture: PyTorch (Dataloader is overrided to enhance multiprocessing performance) and OpenCV

Data Augmentation: image sharpening, rotation, translation, normalization, blur (gaussian, average, median), superpixel, emboss, spatial augmentaion (HSV), convolutional augmentation (not good), elastic transformation (often happen in MNIST dataset), elementwise add, noise, dropout --- imgaug is able to handle all of them

Detection Network: YoloV4 (pending), GRU/BiLSTM (baseline), darknet (baseline, yolov4-tiny, anchor re-organized, mish activation, DIOU NMS, iterations for 10000 rounds, score: 0.833, ranking: top 5%)

Classification Network: pretrained ResNet-50 (Mish/Meta-ACON (CVPR 2021), LeakyRelu) with dropblock/dropout

Optimizer: SGD with Cosine learning rate scheduler

Loss Function: label smoothing loss
