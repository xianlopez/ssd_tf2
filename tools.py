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
        predictions.append(np.take(predictions_nms[i], predictions_nms[i][:, 4] != nclasses, axis=0))
    return predictions


def draw_predictions(img, predictions):
    # img: (height, width, 3)
    # predictions: (num_preds_nms, 6) [xmin, ymin, width, height, class_id, conf]
    height, width, _ = img.shape
    color = (1.0, 0.0, 0.0)
    for i in range(predictions.shape[0]):
        box_xmin_abs = min(max(int(round(predictions[i, 0] * width)), 0), width - 2)
        box_ymin_abs = min(max(int(round(predictions[i, 1] * height)), 0), height - 2)
        box_width_abs = max(int(round(predictions[i, 2] * width)), 1)
        box_height_abs = max(int(round(predictions[i, 3] * height)), 1)
        box_xmax_abs = min(box_xmin_abs + box_width_abs, width - 1)
        box_ymax_abs = min(box_ymin_abs + box_height_abs, height - 1)
        img = cv2.rectangle(img, (box_xmin_abs, box_ymin_abs), (box_xmax_abs, box_ymax_abs), color, thickness=2)
    return img
