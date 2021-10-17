import torch
import numpy as np
from torchvision.ops import nms


def xywh2xyxy(detections):
    if isinstance(detections, torch.Tensor):
        detections = torch.Tensor(list(map(lambda x: [x[0], x[1], x[0]+x[2], x[1]+x[3], x[4]], detections)))
    else:
        detections = np.asarray(list(map(lambda x: [x[0], x[1], x[0]+x[2], x[1]+x[3], x[4]], detections)))
    return detections

def nms_torchvision(detections: torch.Tensor, threshold: float, mode="xywh"):
    if mode == "xywh":
        detections = xywh2xyxy(detections)
    shapes, scores = detections[:, :-1], detections[:, -1]
    return nms(shapes, scores, threshold)

def nms_pytorch(detections: torch.Tensor, threshold1: float, threshold2: float, mode="xywh"):
    """
    :param mode:
    :param detections: (x, y, w, h, confidence)
    :param threshold1: recognized as a valid detection
    :param threshold2: recognized as final detection
    :return:
    """
    if mode == "xywh":
        detections = xywh2xyxy(detections)
    mask = detections[:, -1] > threshold1
    indexes = torch.where(mask)
    detections = detections[indexes]
    shapes, scores = detections[:, :-1], detections[:, -1]
    x1 = shapes[:, 0]
    y1 = shapes[:, 1]
    x2 = shapes[:, 2]
    y2 = shapes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel():
        cur = order[0]
        keep.append(cur)
        xx1 = x1[order[1:]].clamp(min=x1[cur])  # [N-1,]
        yy1 = y1[order[1:]].clamp(min=y1[cur])
        xx2 = x2[order[1:]].clamp(max=x2[cur])
        yy2 = y2[order[1:]].clamp(max=y2[cur])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)  # [N-1,]

        iou = inter / (areas[cur] + areas[order[1:]] - inter)  # [N-1,]
        inds = torch.where(iou <= threshold2)[0]
        order = order[inds + 1]
    return keep


def nms_numpy(detections: np.ndarray, threshold1: float, threshold2: float, mode="xywh"):
    """

    :param mode:
    :param detections: in numpy
    :param threshold1: same as pytorch version
    :param threshold2: same as pytorch version
    :return:
    """
    if detections.size == 0:
        return detections
    if mode == "xywh":
        detections = xywh2xyxy(detections)
    detections = np.asarray(list(filter(lambda x: x[-1] > threshold1, detections)))
    shapes, scores = detections[:, :-1], detections[:, -1]
    x1 = shapes[:, 0]
    y1 = shapes[:, 1]
    x2 = shapes[:, 2]
    y2 = shapes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        cur = order[0]
        keep.append(cur)

        xx1 = np.maximum(x1[cur], x1[order[1:]])
        yy1 = np.maximum(y1[cur], y1[order[1:]])
        xx2 = np.minimum(x2[cur], x2[order[1:]])
        yy2 = np.minimum(y2[cur], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[cur] + areas[order[1:]] - inter)
        inds = np.where(iou <= threshold2)[0]
        order = order[inds + 1]
    return keep


if __name__ == "__main__":
    detections = np.array(
        [[10, 8, 5, 2, 0.4],
        [10, 9, 6, 2, 0.6],
         [9, 8, 6, 2, 0.5],
         [3, 1, 2, 2, 0.7]]
    )
    res = nms_numpy(detections, 0.3, 0.7)
    print(res)
