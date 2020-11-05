import tensorflow as tf
from decoding import decode_coords
from compute_iou import compute_iou_flat
import numpy as np

th_iou_map = 0.5


def mean_ap_on_batch(batch_predictoins, batch_gt_raw, nclasses):
    # batch_predictoins: list with as many elements as batch_size. Each element is like this:
    #                    (npreds, 6) [xmin, ymin, width, height, class_id, conf]
    # batch_gt_raw: (batch_size, max_gt_boxes, 5) [xmin, ymin, width, height, classid)
    batch_size = len(batch_predictoins)
    assert batch_size == batch_gt_raw.shape[0]
    mAPs = np.zeros((batch_size), np.float32)
    for batch_idx in range(batch_size):
        mAPs[batch_idx] = mean_ap_on_image(batch_predictoins[batch_idx], batch_gt_raw[batch_idx, ...], nclasses)
    return np.mean(mAPs)


def mean_ap_on_image(predictions, gt_boxes_raw, nclasses):
    # predictions: (npreds, 6) [xmin, ymin, width, height, class_id, conf]
    # gt_boxes_raw: (max_gt_boxes, 5) [xmin, ymin, width, height, classid)
    APs = np.zeros((nclasses), np.float32)
    for class_id in range(nclasses):
        preds_this_class = np.take(predictions, predictions[:, 4] == class_id, axis=0)
        gt_this_class = np.take(gt_boxes_raw, gt_boxes_raw[:, 4] == class_id, axis=0)
        APs[class_id] = compute_ap(preds_this_class, gt_this_class)
    return np.mean(APs)


def compute_ap(preds, gt):
    # preds: (npreds_this_class, 6) [xmin, ymin, width, height, class_id, conf]
    # gt: (ngt_this_class, 5) [xmin, ymin, width, height, class_id)
    # Note: classes ids are not checked in this function. We assume all predictions
    # and GT boxes belong to the same class.
    npreds_this_class = preds.shape[0]
    ngt_this_class = gt.shape[0]
    if ngt_this_class == 0:
        return 0.0
    elif npreds_this_class == 0:
        return 0.0
    else:
        # Compute matches between predictions and ground truth:
        iou = compute_iou_flat(preds[:, :4], gt[:, :4])  # (npreds_this_class, ngt_this_class)
        sorted_preds_idxs = np.argsort(-preds[:, 5])
        available_gt = np.ones((ngt_this_class), np.bool)
        matched_preds = np.zeros((npreds_this_class), np.bool)
        for pred_idx in sorted_preds_idxs:
            # print('pred_idx = ' + str(pred_idx))
            sorted_gt_idxs = np.argsort(-iou[pred_idx, :])
            for gt_idx in sorted_gt_idxs:
                # print('gt_idx = ' + str(gt_idx))
                if iou[pred_idx, gt_idx] > th_iou_map:
                    # print('iou ok')
                    if available_gt[gt_idx]:
                        # print('gt available')
                        available_gt[gt_idx] = False
                        matched_preds[pred_idx] = True
                        break
                else:
                    break

        # Compute precision-recall curve:
        precision = np.zeros((npreds_this_class), np.float32)
        recall = np.zeros((npreds_this_class), np.float32)
        TP = 0
        FP = 0
        count = 0
        for pred_idx in sorted_preds_idxs:
            if matched_preds[pred_idx]:
                TP += 1
            else:
                FP += 1
            precision[count] = TP / float(TP + FP)
            recall[count] = TP / float(ngt_this_class)
            count += 1
        # Rectify precision:
        max_to_the_right = 0
        for i in range(npreds_this_class - 1, -1, -1):
            max_to_the_right = max(precision[i], max_to_the_right)
            precision[i] = max_to_the_right

        # Integrate precision-recall curve:
        AP = precision[0] * recall[0]
        for i in range(1, npreds_this_class):
            AP += precision[i] * (recall[i] - recall[i - 1])

        return AP







