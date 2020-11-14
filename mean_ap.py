import tensorflow as tf
from decoding import decode_coords, decode_preds
from compute_iou import compute_iou_flat
import numpy as np

th_iou_map = 0.5


def compute_map(all_batches_net_output, all_batches_gt_raw, anchors):
    # all_batches_net_output: List with as many elements as the number of batches. Each element has this form:
    # (batch_size, nanchors, 4 + nclasses + 1). The spatial coordinates are encoded.
    # all_batches_gt_raw: List with as many elements as the number of batches. Each element has this form:
    # (batch_size, max_gt_boxes, 5). The spatial coordinates are not encoded [xmin, ymin, width, height].
    # anchors: (nanchors, 4) [xmin, ymin, width, height]
    assert len(all_batches_net_output) == len(all_batches_gt_raw)
    nbatches = len(all_batches_net_output)
    if nbatches == 0:
        return 0.0
    assert len(all_batches_net_output[0].shape) == 3
    assert len(all_batches_gt_raw[0].shape) == 3
    assert all_batches_net_output[0].shape[0] == all_batches_gt_raw[0].shape[0]
    batch_size = all_batches_net_output[0].shape[0]
    assert all_batches_gt_raw[0].shape[2] == 5
    nclasses = all_batches_net_output[0].shape[2] - 5
    assert nclasses > 0
    assert all_batches_net_output[0].shape[1] == anchors.shape[0]

    num_gts_each_class = np.zeros((nclasses), np.int32)
    npreds_total = 0
    # These will be lists with one element per image:
    all_matched_flags = []
    all_confs = []
    all_class_ids = []
    for batch_idx in range(nbatches):
        predictions = decode_preds(all_batches_net_output[batch_idx], anchors, nclasses)  # List of (npreds, 6)
        for idx_in_batch in range(batch_size):
            matched_flags = match_predictions_on_image(
                predictions[idx_in_batch], all_batches_gt_raw[batch_idx][idx_in_batch, :, :], nclasses)  # (npreds)
            all_matched_flags.append(matched_flags)
            all_confs.append(predictions[idx_in_batch][:, 5])
            all_class_ids.append(predictions[idx_in_batch][:, 4])
            npreds_total += matched_flags.shape[0]
            for gt_idx in range(all_batches_gt_raw[batch_idx].shape[1]):
                gt_class_id = int(round(all_batches_gt_raw[batch_idx][idx_in_batch, gt_idx, 4]))
                if gt_class_id < nclasses:
                    num_gts_each_class[gt_class_id] += 1
                else:
                    break  # We assume the positive GTs are at the beginning

    # Flatten flags, confs and class ids into 1D numpy arrays:
    matched_flags_flat = np.zeros((npreds_total), np.bool)
    confs_flat = np.zeros((npreds_total), np.float64)
    class_ids_flat = np.zeros((npreds_total), np.float64)
    start_idx = 0
    for img_idx in range(len(all_matched_flags)):
        npreds_img = all_matched_flags[img_idx].shape[0]
        end_idx = start_idx + npreds_img
        matched_flags_flat[start_idx:end_idx] = all_matched_flags[img_idx]
        confs_flat[start_idx:end_idx] = all_confs[img_idx]
        class_ids_flat[start_idx:end_idx] = all_class_ids[img_idx]
        start_idx = end_idx

    APs = np.zeros((nclasses), np.float64)
    for class_id in range(nclasses):
        indices = np.where(class_ids_flat == class_id)[0]
        if len(indices) == 0:
            continue
        flags_this_class = np.take(matched_flags_flat, indices)
        confs_this_class = np.take(confs_flat, indices)
        APs[class_id] = compute_ap(confs_this_class, flags_this_class, num_gts_each_class[class_id])

    mAP = np.mean(APs)
    return mAP


def compute_ap(confidences, matched_flags, ngt_this_class):
    # confidences: (npreds_this_class)
    # matched_flags: (npreds_this_class)
    # We assume here all the predictions belong to the same class.
    assert len(confidences) == len(matched_flags)
    npreds_this_class = len(confidences)
    sorted_conf_idxs = np.argsort(-confidences)

    # Compute precision-recall curve:
    precision = np.zeros((npreds_this_class), np.float32)
    recall = np.zeros((npreds_this_class), np.float32)
    TP = 0
    FP = 0
    count = 0
    for pred_idx in sorted_conf_idxs:
        if matched_flags[pred_idx]:
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


def match_predictions_on_image(predictions, gt_boxes_raw, nclasses):
    # predictions: (npreds, 6) [xmin, ymin, width, height, class_id, conf]
    # gt_boxes_raw: (max_gt_boxes, 5) [xmin, ymin, width, height, classid)
    matched_flags = np.zeros((predictions.shape[0]), np.bool)
    for class_id in range(nclasses):
        pred_indexes_this_class = np.where(predictions[:, 4] == class_id)[0]
        preds_this_class = np.take(predictions, pred_indexes_this_class, axis=0)
        gt_this_class = np.take(gt_boxes_raw, np.where(gt_boxes_raw[:, 4] == class_id)[0], axis=0)
        matched_flags_this_class = match_predictions_same_class(preds_this_class, gt_this_class)
        matched_flags[pred_indexes_this_class] = matched_flags_this_class
    return matched_flags


def match_predictions_same_class(preds, gt):
    # preds: (npreds_this_class, 6) [xmin, ymin, width, height, class_id, conf]
    # gt: (ngt_this_class, 5) [xmin, ymin, width, height, class_id)
    # Note: classes ids are not checked in this function. We assume all predictions
    # and GT boxes belong to the same class.
    npreds_this_class = preds.shape[0]
    ngt_this_class = gt.shape[0]
    matched_flags = np.zeros((npreds_this_class), np.bool)
    if ngt_this_class > 0 and npreds_this_class > 0:
        iou = compute_iou_flat(preds[:, :4], gt[:, :4])  # (npreds_this_class, ngt_this_class)
        sorted_preds_idxs = np.argsort(-preds[:, 5])
        available_gt = np.ones((ngt_this_class), np.bool)
        for pred_idx in sorted_preds_idxs:
            sorted_gt_idxs = np.argsort(-iou[pred_idx, :])
            for gt_idx in sorted_gt_idxs:
                if iou[pred_idx, gt_idx] > th_iou_map:
                    if available_gt[gt_idx]:
                        available_gt[gt_idx] = False
                        matched_flags[pred_idx] = True
                        break
                else:
                    break
    return matched_flags

