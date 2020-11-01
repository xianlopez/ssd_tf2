import numpy as np


def keep_best_class(net_output):
    # net_output: (batch_size, nanchors, 4 + nclasses + 1)
    print('keep_best_class')
    print('net_output.shape: ' + str(net_output.shape))
    class_id = np.expand_dims(np.argmax(net_output[:, :, 4:], axis=-1), axis=-1)  # (batch_size, nanchors, 1)
    print('class_id.shape: ' + str(class_id.shape))
    # Note: What is better, take_along_axis or max?
    conf = np.take_along_axis(net_output[:, :, 4:], class_id, axis=-1)  # (batch_size, nanchors, 1)
    print('conf.shape: ' + str(conf.shape))
    return np.concatenate([net_output[:, :, :4], class_id, conf], axis=-1)  # (batch_size, nanchors, 6)


def remove_background_predictions(predictions_full, nclasses):
    # predictions_full: (batch_size, nanchors, 6) [xmin, ymin, width, height, class_id, conf]
    return np.take(predictions_full, predictions_full[:, :, 4] != nclasses, axis=0)
