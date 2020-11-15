import numpy as np
import os


def delete_checkpoint_with_index(index):
    prefix = 'ckpts/ckpt_' + str(index) + '.'
    folder = os.path.dirname(prefix)
    prefix_only_name = os.path.basename(prefix)
    assert os.path.isdir(folder)
    for name in os.listdir(folder):
        if prefix_only_name in name:
            file_path = os.path.join(folder, name)
            os.remove(file_path)


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
