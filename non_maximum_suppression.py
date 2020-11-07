import numpy as np
from compute_iou import compute_iou_flat

th_nms = 0.5


def batch_non_maximum_suppression_slow(predictions_full, nclasses):
    # predictions_full: (batch_size, nanchors, 6) [xmin, ymin, width, height, class_id, conf]
    predictions_full_nms = np.zeros_like(predictions_full)
    for i in range(predictions_full.shape[0]):
        predictions_full_nms[i, :, :] = non_maximum_suppression_slow(predictions_full[i, :, :], nclasses)
    return predictions_full_nms


def batch_non_maximum_suppression_fast(predictions_full, nclasses):
    # predictions_full: (batch_size, nanchors, 6) [xmin, ymin, width, height, class_id, conf]
    predictions_nms = []
    for i in range(predictions_full.shape[0]):
        predictions_nms.append(non_maximum_suppression_fast(predictions_full[i, :, :], nclasses))
    return predictions_nms


def non_maximum_suppression_slow(predictions_full, nclasses):
    # predictions_full: (nanchors, 6) [xmin, ymin, width, height, class_id, conf]
    assert len(predictions_full.shape) == 2
    coords = predictions_full[:, :4]
    iou = compute_iou_flat(coords, coords)
    nanchors = predictions_full.shape[0]
    sorted_indexes = np.argsort(predictions_full[:, 5])
    for i in range(nanchors):
        class_i = predictions_full[sorted_indexes[i], 4]
        for j in range(i + 1, nanchors):
            if iou[sorted_indexes[i], sorted_indexes[j]] > th_nms:
                if predictions_full[sorted_indexes[j], 4] == class_i:
                    predictions_full[sorted_indexes[i], 4] = nclasses  # Assign background
                    break
    # Note: This function doesn't return anything, because predictions_full is modified in place.


def non_maximum_suppression_fast(predictions_full, nclasses):
    # predictions_full: (nanchors, 6) [xmin, ymin, width, height, class_id, conf]
    assert len(predictions_full.shape) == 2

    if len(predictions_full) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    remaining_preds = []
    for class_id in range(nclasses):
        preds_this_class = predictions_full[np.where(np.abs(predictions_full[:, 4] - class_id) < 0.5)[0], :]
        pick = non_maximum_suppression_fast_on_class(preds_this_class)
        preds_this_class = preds_this_class[pick]
        remaining_preds.append(preds_this_class)

    result = np.concatenate(remaining_preds, axis=0)  # (num_preds_nms, 6)

    return result


def non_maximum_suppression_fast_on_class(boxes):
    # boxes: (npreds_class, 4) [xmin, ymin, width, height, class_id, conf]
    # Note 1: strongly based on this:
    # https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    # The main modifications are the use of IOU instead of their 'overlap', and that I
    # sort the boxes by confidence.
    # Note 2: This kind of non-maximum suppression doesn't allow for "chain" suppression.
    assert len(boxes.shape) == 2
    assert boxes.shape[1] == 6

    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    area = boxes[:, 2] * boxes[:, 3]

    idxs = np.argsort(boxes[:, 5])

    pick = []
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        ww = np.maximum(0, xx2 - xx1)
        hh = np.maximum(0, yy2 - yy1)

        intersection = ww * hh
        union = area[i] + area[idxs[:last]] - intersection
        iou = intersection / union

        idxs = np.delete(idxs, np.concatenate(([last], np.where(iou > th_nms)[0])))

    return pick
