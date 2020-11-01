import numpy as np

max_detections_per_image = 50


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


def decode_preds(net_output, anchors, th_conf):
    # net_output: (nanchors, 4 + nclasses + 1)
    #       [xc_enc, yc_enc, width_enc, height_enc, prob_class1, ..., prob_classN, prob_background]
    # anchors: (nanchors, 4) [xmin, ymin, width, height]
    # Note we don't consider the batch dimension here.
    classes_probs = net_output[:, 4:-1]
    max_positive_conf_all_boxes = np.max(classes_probs, axis=1)
    sorted_indexes = np.argsort(-max_positive_conf_all_boxes)
    indices_keep = sorted_indexes[:max_detections_per_image]
    predictions = []
    for idx in indices_keep:
        if max_positive_conf_all_boxes[idx] > th_conf:
            xmin, ymin, width, height = decode_box(net_output[idx, :4], anchors[idx, :])
            class_id = np.argmax(classes_probs[idx, :])
            box = PredictedBox([xmin, ymin, width, height], class_id, max_positive_conf_all_boxes[idx])
            predictions.append(box)
    return predictions


def decode_coords(boxes_enc, anchors):
    # boxes_enc: (nanchors, 4) [xc_enc, yc_enc, width_enc, height_enc]
    # anchors: (nanchors, 4) [xmin, ymin, width, height]

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
