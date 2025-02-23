# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    # 函数接受两个参数 boxes1 和 boxes2，这些应该是表示边界框的张量
    # 利用了PyTorch的张量操作和广播机制，可以同时处理多个边界框，非常适合在目标检测等任务中使用。
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    # 计算每对边界框的交集区域的左上角和右下角坐标：
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    # 计算交集区域的宽度和高度，并确保它们不为负：
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    # 计算交集面积：
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    # 计算并集面积：
    union = area1[:, None] + area2 - inter
    # 计算交集面积与并集面积的比率：
    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    广义IoU的计算公式为：GIoU = IoU - (C - union) / C，其中C是最小闭包矩形的面积。

    这个函数实现了广义IoU，它比普通IoU更好地处理非重叠情况，并提供了一个更平滑的梯度。这在目标检测和物体定位任务中特别有用，可以帮助模型更好地学习边界框的位置和大小。
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    # 进行断言检查，确保边界框是有效的（右下角坐标大于等于左上角坐标）：
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    # 调用 box_iou 函数计算普通的IoU和并集面积：
    iou, union = box_iou(boxes1, boxes2)
    # 计算包含两个边界框的最小闭包矩形的左上角和右下角坐标：
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    # 计算最小闭包矩形的宽度和高度，并确保它们不为负：
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    # 计算最小闭包矩形的面积：
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    # 函数接受一个参数masks，这是一个形状为[N, H, W]的张量，其中N是掩码数量，H和W是空间维度。
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    这个函数的主要目的是找出每个掩码中非零区域的最小外接矩形，并将其表示为边界框。它通过巧妙地使用坐标网格和掩码的乘法操作，高效地计算出每个掩码的边界。
    返回的边界框格式为[x_min, y_min, x_max, y_max]，这是目标检测中常用的xyxy格式。
    """
    # 检查掩码是否为空
    if masks.numel() == 0:# 如果为空，返回一个空的边界框张量。
        return torch.zeros((0, 4), device=masks.device)
    # 获取掩码的高度和宽度：
    h, w = masks.shape[-2:]
    # 创建坐标网格：
    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    # 计算掩码的边界框：
    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
