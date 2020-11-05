import numpy as np


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
