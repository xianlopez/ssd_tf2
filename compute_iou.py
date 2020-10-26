import numpy as np


def compute_iou_flat(boxes1, boxes2):
    # boxes1 (nboxes1, 4) [xmin, ymin, width, height]
    # boxes2 (nboxes2, 4) [xmin, ymin, width, height]
    nboxes1 = boxes1.shape[0]
    nboxes2 = boxes2.shape[0]
    boxes1_expanded = np.expand_dims(boxes1, axis=1)  # (nboxes1, 1, 4)
    boxes1_expanded = np.tile(boxes1_expanded, (1, nboxes2, 1))  # (nboxes1, nboxes2, 4)
    boxes2_expanded = np.expand_dims(boxes2, axis=0)  # (1, nboxes2, 4)
    boxes2_expanded = np.tile(boxes2_expanded, (nboxes1, 1, 1))  # (nboxes1, nboxes2, 4)
    xmin = np.max(np.stack((boxes1_expanded[:, :, 0], boxes2_expanded[:, :, 0]), axis=2), axis=2)  # (nboxes1, nboxes2)
    ymin = np.max(np.stack((boxes1_expanded[:, :, 1], boxes2_expanded[:, :, 1]), axis=2), axis=2)  # (nboxes1, nboxes2)
    xmax = np.min(np.stack((boxes1_expanded[:, :, 0] + boxes1_expanded[:, :, 2],
                            boxes2_expanded[:, :, 0] + boxes2_expanded[:, :, 2]), axis=2), axis=2)  # (nboxes1, nboxes2)
    ymax = np.min(np.stack((boxes1_expanded[:, :, 1] + boxes1_expanded[:, :, 3],
                            boxes2_expanded[:, :, 1] + boxes2_expanded[:, :, 3]), axis=2), axis=2)  # (nboxes1, nboxes2)
    zero_grid = np.zeros((nboxes1, nboxes2))
    w = np.max(np.stack((xmax - xmin, zero_grid), axis=2), axis=2)  # (nboxes1, nboxes2)
    h = np.max(np.stack((ymax - ymin, zero_grid), axis=2), axis=2)  # (nboxes1, nboxes2)
    area_inter = w * h  # (nboxes1, nboxes2)
    area_anchors = boxes2_expanded[:, :, 2] * boxes2_expanded[:, :, 3]  # (nboxes1, nboxes2)
    area_boxes = boxes1_expanded[:, :, 2] * boxes1_expanded[:, :, 3]  # (nboxes1, nboxes2)
    area_union = area_anchors + area_boxes - area_inter  # (nboxes1, nboxes2)
    iou = area_inter / area_union  # (nboxes1, nboxes2)
    return iou  # (nboxes1, nboxes2)


def compute_iou_single(box1, box2):
    # box coordinates: [xmin, ymin, w, h]
    assert np.all(box1[2:] >= 0.0)
    assert np.all(box2[2:] >= 0.0)
    lu = np.max(np.array([box1[:2], box2[:2]]), axis=0)
    rd = np.min(np.array([[box1[0] + box1[2], box1[1] + box1[3]], [box2[0] + box2[2], box2[1] + box2[3]]]), axis=0)
    intersection = np.maximum(0.0, rd-lu)
    intersec_area = intersection[0] * intersection[1]
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    if area1 < 1e-6 and area2 < 1e-6:
        iou = 0.0
    else:
        union_area = area1 + area2 - intersec_area
        iou = intersec_area / np.float(union_area)
    return iou
