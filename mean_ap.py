# import tensorflow as tf
#
#
# def mean_ap(y_true, y_pred):
#         # Input format (both y_true and y_pred):
#         # (?, nanchors, 4 + nclasses + 1) [xmin, ymin, width, height, class1, ..., classN, background]
#
#         output_boxes =
#         selected_indices = tf.image.non_max_suppression(
#             boxes, scores, max_output_size, iou_threshold)
#         selected_boxes = tf.gather(boxes, selected_indices)
