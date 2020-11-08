import numpy as np
import cv2


def keep_best_class(net_output):
    # net_output: (batch_size, nanchors, 4 + nclasses + 1)
    class_id = np.expand_dims(np.argmax(net_output[:, :, 4:], axis=-1), axis=-1)  # (batch_size, nanchors, 1)
    # Note: What is better, take_along_axis or max?
    conf = np.take_along_axis(net_output[:, :, 4:], class_id, axis=-1)  # (batch_size, nanchors, 1)
    return np.concatenate([net_output[:, :, :4], class_id, conf], axis=-1)  # (batch_size, nanchors, 6)


def remove_background_predictions(predictions_nms, nclasses):
    # predictions_nms: List with one element per image. Each element is like this:
    # (num_preds_nms, 6) [xmin, ymin, width, height, class_id, conf]
    predictions = []
    for i in range(len(predictions_nms)):
        predictions.append(np.take(predictions_nms[i], np.where(predictions_nms[i][:, 4] != nclasses)[0], axis=0))
    return predictions


def draw_boxes(img, boxes):
    # img: (height, width, 3)
    # boxes: list of boxes. Each box comes as [xmin, ymin, width, height].
    height, width, _ = img.shape
    color = (1.0, 0.0, 0.0)
    for i in range(len(boxes)):
        box_xmin_abs = min(max(int(round(boxes[i][0] * width)), 0), width - 2)
        box_ymin_abs = min(max(int(round(boxes[i][1] * height)), 0), height - 2)
        box_width_abs = max(int(round(boxes[i][2] * width)), 1)
        box_height_abs = max(int(round(boxes[i][3] * height)), 1)
        box_xmax_abs = min(box_xmin_abs + box_width_abs, width - 1)
        box_ymax_abs = min(box_ymin_abs + box_height_abs, height - 1)
        img = cv2.rectangle(img, (box_xmin_abs, box_ymin_abs), (box_xmax_abs, box_ymax_abs), color, thickness=2)
    return img


def display_anchors(img, anchors):
    # anchors: (nanchors, 4) [xmin, ymin, width, height]
    height, width, _ = img.shape
    color = (0.0, 1.0, 1.0)
    # for i in range(2000, anchors.shape[0]):
    for i in range(anchors.shape[0]):
        img_to_show = img.copy()
        anc_xmin_abs = int(round(anchors[i, 0] * width))
        anc_ymin_abs = int(round(anchors[i, 1] * height))
        anc_width_abs = int(round(anchors[i, 2] * width))
        anc_height_abs = int(round(anchors[i, 3] * height))
        anc_xmax_abs = anc_xmin_abs + anc_width_abs
        anc_ymax_abs = anc_ymin_abs + anc_height_abs
        img_to_show = cv2.rectangle(img_to_show, (anc_xmin_abs, anc_ymin_abs),
                                    (anc_xmax_abs, anc_ymax_abs), color, thickness=2)
        cv2.imshow('anchors', img_to_show)
        cv2.waitKey(100)
