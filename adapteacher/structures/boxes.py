# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by XuDong Wang from https://github.com/facebookresearch/detectron2/blob/main/detectron2/structures/boxes.py

import torch

def pairwise_iou_max_scores(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M, compute the IoU
    (intersection over union) between **all** N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    # area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N]
    # area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M]

    # width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
    #     boxes1[:, None, :2], boxes2[:, :2]
    # )  # [N,M,2]

    # width_height.clamp_(min=0)  # [N,M,2]
    # inter = width_height.prod(dim=2)  # [N,M]

    # # handle empty boxes
    # iou = torch.where(
    #     inter > 0,
    #     inter / (area1[:, None] + area2 - inter),
    #     torch.zeros(1, dtype=inter.dtype, device=inter.device),
    # )
    # iou_max, _ = torch.max(iou, dim=1)
    # return iou_max

    # Initialize tensor to store max IoU for each box in boxes1 for each class
    num_classes = boxes1.shape[1] // 4
    max_iou_per_class = torch.zeros((boxes1.shape[0], num_classes), device=boxes1.device)

    # Calculate IoU for each class separately
    for cls in range(num_classes):
        start_idx = cls * 4
        end_idx = start_idx + 4

        cls_boxes1 = boxes1[:, start_idx:end_idx]

        area1 = (cls_boxes1[:, 2] - cls_boxes1[:, 0]) * (cls_boxes1[:, 3] - cls_boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        width_height = torch.min(cls_boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(cls_boxes1[:, None, :2], boxes2[:, :2])
        width_height.clamp_(min=0)
        inter = width_height.prod(dim=2)

        iou = torch.where(inter > 0, inter / (area1[:, None] + area2 - inter), torch.zeros(1, dtype=inter.dtype, device=inter.device))
        iou_max, _ = torch.max(iou, dim=1)

        max_iou_per_class[:, cls] = iou_max # shape: [N, num_classes]

    max_iou, _ = torch.max(max_iou_per_class, dim=1)
    return max_iou