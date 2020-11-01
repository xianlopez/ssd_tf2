import numpy as np
from compute_iou import compute_iou_flat
from decoding import decode_coords

th_nms = 0.5


def batch_non_maximum_suppression(predictions_full, nclasses):
    # predictions_full: (batch_size, nanchors, 6) [xmin, ymin, width, height, class_id, conf]
    predictions_full_nms = np.zeros_like(predictions_full)
    for i in range(predictions_full.shape[0]):
        predictions_full_nms[i, :, :] = non_maximum_suppression(predictions_full[i, :, :], nclasses)
    return predictions_full_nms


def non_maximum_suppression(predictions_full, nclasses):
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
