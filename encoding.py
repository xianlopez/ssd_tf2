import numpy as np
from compute_iou import compute_iou_flat

th_match = 0.5


def encode_gt(boxes, anchors, nclasses):
    # boxes: (nboxes, 5) [xmin, ymin, width, height, classid]
    # anchors: (nanchors, 4)

    n_gt = boxes.shape[0]
    nanchors = anchors.shape[0]

    localizations_flat = np.zeros(shape=(nanchors, 4), dtype=np.float32)
    labels_onehot_flat = np.zeros(shape=(nanchors, nclasses + 1), dtype=np.float32)

    gt_vec = boxes[:, :4]
    iou = compute_iou_flat(gt_vec, anchors)  # (n_gt, nanchors)

    # TODO: Why this?
    max_iou_per_gt = np.max(iou, axis=1)  # (n_gt)
    for gt_idx in range(n_gt):
        for anc_idx in range(nanchors):
            if np.abs(iou[gt_idx, anc_idx] - max_iou_per_gt[gt_idx]) < 1e-4:
                iou[gt_idx, anc_idx] = max_iou_per_gt[gt_idx]

    correspondences = -np.ones(shape=(nanchors), dtype=np.int32)
    if n_gt > 0:
        # First iterate the ground truth boxes, starting with the ones that have most IOU,
        # and assign them to the anchor with highest IOU.
        indices = np.argsort(-max_iou_per_gt)
        for gt_idx in indices:
            found_here = False

            # TODO: I think I can delete this block:
            for anc_idx in range(nanchors):
                if np.abs(iou[gt_idx, anc_idx] - max_iou_per_gt[gt_idx]) < 1e-4:
                    if correspondences[anc_idx] < 0:
                        correspondences[anc_idx] = gt_idx
                        found_here = True
                        break

            if not found_here:
                anchors_indices = np.argsort(-iou[gt_idx, :])
                for anc_idx in anchors_indices:
                    if correspondences[anc_idx] < 0:
                        correspondences[anc_idx] = gt_idx
                        break

        # Then iterate the remaining anchors, and assign them to every gt box that has an IOU over the threshold:
        # TODO: I think this is not exactly what I've just described...
        for anc_idx in range(nanchors):
            if correspondences[anc_idx] < 0:
                gt_idx = np.argmax(iou[:, anc_idx])
                if iou[gt_idx, anc_idx] > th_match:
                    correspondences[anc_idx] = gt_idx

    for anc_idx, gt_idx in enumerate(correspondences):
        if gt_idx >= 0:
            xc_enc, yc_enc, w_enc, h_enc = encode_box(boxes[gt_idx, :4], anchors[anc_idx])
            localizations_flat[anc_idx, :] = [xc_enc, yc_enc, w_enc, h_enc]
            labels_onehot_flat[anc_idx, int(boxes[gt_idx, 4])] = 1.0
        else:
            labels_onehot_flat[anc_idx, -1] = 1.0

    encoded_labels = np.concatenate((localizations_flat, labels_onehot_flat), axis=1)

    return encoded_labels  # (nanchors, 4 + nclasses + 1)


def encode_box(box_coords, anchor):
    # box_coords, anchor: [xmin, ymin, width, height]
    # The output is the encoded box (which has offsets with respect to the center and to the width and height)
    box_xmin = box_coords[0]
    box_ymin = box_coords[1]
    box_width = box_coords[2]
    box_height = box_coords[3]
    box_xc = box_xmin + box_width / 2.0
    box_yc = box_ymin + box_height / 2.0

    anchor_xc = anchor[0] + anchor[2] / 2.0
    anchor_yc = anchor[1] + anchor[3] / 2.0
    anchor_width = anchor[2]
    anchor_height = anchor[3]

    xc_enc = (box_xc - anchor_xc) / anchor_width  # x (center)
    yc_enc = (box_yc - anchor_yc) / anchor_height  # y (center)
    w_enc = np.log(box_width / anchor_width)  # width
    h_enc = np.log(box_height / anchor_height)  # height

    # Apply 'variances':
    xc_enc /= 0.1
    yc_enc /= 0.1
    w_enc /= 0.2
    h_enc /= 0.2

    return xc_enc, yc_enc, w_enc, h_enc
