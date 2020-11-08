import numpy as np
import tools
from non_maximum_suppression import batch_non_maximum_suppression_fast
import scipy


def decode_box(box_enc, anchor):
    # box_enc: Encoded box, with offsets with respect to the center and to the width and height.
    # anchor: [xmin, ymin, width, height]
    xc_enc = box_enc[0]
    yc_enc = box_enc[1]
    w_enc = box_enc[2]
    h_enc = box_enc[3]

    # Take into account the 'variances':
    xc_enc *= 0.1
    yc_enc *= 0.1
    w_enc *= 0.2
    h_enc *= 0.2

    anchor_xc = anchor[0] + anchor[2] / 2.0
    anchor_yc = anchor[1] + anchor[3] / 2.0
    anchor_width = anchor[2]
    anchor_height = anchor[3]

    xc = xc_enc * anchor_width + anchor_xc
    yc = yc_enc * anchor_height + anchor_yc
    width = np.exp(w_enc) * anchor_width
    height = np.exp(h_enc) * anchor_height
    xmin = xc - width / 2.0
    ymin = yc - height / 2.0

    # TODO: Do I need this?
    # Ensure the box fits in the image:
    xmin = min(max(xmin, 0), 1)
    ymin = min(max(ymin, 0), 1)
    width = min(max(width, 0), 1 - xmin)
    height = min(max(height, 0), 1 - ymin)

    return xmin, ymin, width, height


def decode_preds(net_output, anchors, nclasses):
    # net_output:  (batch_size, nanchors, 4 + nclasses + 1)
    coords_enc = net_output[:, :, :4]
    logits = net_output[:, :, 4:]
    confs = scipy.special.softmax(logits, axis=-1)  # (batch_size, nanchors, nclasses + 1)
    coords_dec = decode_coords_batch(coords_enc, anchors)  # (batch_size, nanchors, 4)
    net_output_dec = np.concatenate([coords_dec, confs], axis=-1)
    predictions_full = tools.keep_best_class(net_output_dec)  # (batch_size, nanchors, 6)
    predictions_nms = batch_non_maximum_suppression_fast(predictions_full, nclasses)
    predictions = tools.remove_background_predictions(predictions_nms, nclasses)
    return predictions


def decode_coords_batch(boxes_enc, anchors):
    # boxes_enc: (batch_size, nanchors, 4) [xc_enc, yc_enc, width_enc, height_enc]
    # anchors: (nanchors, 4) [xmin, ymin, width, height]

    assert len(boxes_enc.shape) == 3

    anchors_exp = np.expand_dims(anchors, axis=0)  # (1, nanchors, 4)

    xc_enc = boxes_enc[..., 0]  # (batch_size, nanchors)
    yc_enc = boxes_enc[..., 1]
    w_enc = boxes_enc[..., 2]
    h_enc = boxes_enc[..., 3]

    # Take into account the 'variances':
    xc_enc *= 0.1
    yc_enc *= 0.1
    w_enc *= 0.2
    h_enc *= 0.2

    anchor_xc = anchors_exp[..., 0] + anchors_exp[..., 2] / 2.0
    anchor_yc = anchors_exp[..., 1] + anchors_exp[..., 3] / 2.0
    anchor_width = anchors_exp[..., 2]
    anchor_height = anchors_exp[..., 3]  # (1, nanchors)

    xc = xc_enc * anchor_width + anchor_xc
    yc = yc_enc * anchor_height + anchor_yc
    width = np.exp(w_enc) * anchor_width
    height = np.exp(h_enc) * anchor_height
    xmin = xc - width / 2.0
    ymin = yc - height / 2.0

    boxes_dec = np.stack([xmin, ymin, width, height], axis=-1)

    assert boxes_dec.shape == boxes_enc.shape

    return boxes_dec  # (batch_size, nanchors, 4) [xmin, ymin, width, height]


def decode_coords(boxes_enc, anchors):
    # boxes_enc: (nanchors, 4) [xc_enc, yc_enc, width_enc, height_enc]
    # anchors: (nanchors, 4) [xmin, ymin, width, height]

    assert len(boxes_enc.shape) == 2

    xc_enc = boxes_enc[:, 0]
    yc_enc = boxes_enc[:, 1]
    w_enc = boxes_enc[:, 2]
    h_enc = boxes_enc[:, 3]

    # Take into account the 'variances':
    xc_enc *= 0.1
    yc_enc *= 0.1
    w_enc *= 0.2
    h_enc *= 0.2

    anchor_xc = anchors[:, 0] + anchors[:, 2] / 2.0
    anchor_yc = anchors[:, 1] + anchors[:, 3] / 2.0
    anchor_width = anchors[:, 2]
    anchor_height = anchors[:, 3]

    xc = xc_enc * anchor_width + anchor_xc
    yc = yc_enc * anchor_height + anchor_yc
    width = np.exp(w_enc) * anchor_width
    height = np.exp(h_enc) * anchor_height
    xmin = xc - width / 2.0
    ymin = yc - height / 2.0

    boxes_dec = np.stack([xmin, ymin, width, height], axis=1)

    return boxes_dec  # (nanchors, 4) [xmin, ymin, width, height]


def decode_gt_batch(batch_gt, anchors):
    # batch_gt: (batch_size, nanchors, 4 + nclasses + 1)
    # [xmin_enc, ymin_enc, width_enc, height_enc, class1, ..., classN, background]
    # anchors: (nanchors, 4) [xmin, ymin, width, height]
    coords_enc = batch_gt[:, :, :4]
    coords_dec = decode_coords_batch(coords_enc, anchors)  # (batch_size, nanchors, 4)
    class_id = np.expand_dims(np.argmax(batch_gt[:, :, 4:], axis=-1), axis=-1)  # (batch_size, nanchors, 1)
    gt_dec = np.concatenate([coords_dec, class_id], axis=-1)
    nclasses = batch_gt.shape[2] - 5
    positive_gt = []
    for i in range(gt_dec.shape[0]):
        positive_gt_i = np.take(gt_dec[i, :, :], np.where(gt_dec[i, :, 4] != nclasses)[0], axis=0)  # (ngt_i, 5)
        # print('ngt_i = ' + str(positive_gt_i.shape[0]))
        # Remove duplicated:
        positive_gt_i_reduced = []
        for j in range(positive_gt_i.shape[0]):
            new_box = positive_gt_i[j, :]
            repeated = False
            for box in positive_gt_i_reduced:
                if np.max(np.abs(box[:4] - new_box[:4])) < 0.0001:
                    repeated = True
                    break
            if not repeated:
                positive_gt_i_reduced.append(new_box)
        # print('ngt_i_reduced = ' + str(len(positive_gt_i_reduced)))
        positive_gt.append(np.array(positive_gt_i_reduced, np.float32))
    return positive_gt  # list with elements of shape (ngt_i_reduced, 5)
