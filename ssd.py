import tensorflow as tf
import numpy as np
from BoundingBoxes import BoundingBox, PredictedBox, check_duplicated

from tensorflow.keras import Model, layers

########## SSD CONFIG ###########
class SSDConfig:
    th_iou_assign_gt = 0.5
    # feat_layers_names = ['block4', 'block7', 'block8', 'block9', 'block10', 'block11']
    feat_layers_names = ['conv4_3', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2']
    # feat_layers_names = ['conv4', 'fc7', 'conv6', 'conv7', 'conv8', 'conv9']
    # anchor_sizes = [21, 45, 99, 153, 207, 261]
    # anchor_sizes = [30, 60, 111, 162, 213, 264, 315]
    anchor_sizes = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]
    steps = [8, 16, 32, 64, 100, 300]
    anchor_ratios = [[1, 2, .5],
                    [1, 2, .5, 3, 1. / 3],
                    [1, 2, .5, 3, 1. / 3],
                    [1, 2, .5, 3, 1. / 3],
                    [1, 2, .5],
                    [1, 2, .5]]
    grid_sizes = [38, 19, 10, 5, 3, 1]
    th_match = 0.5
    image_size = 300
    negative_ratio = 3
    loss_conf_factor = 1
    use_background_class = False
    max_detections_per_image = 50
##################################


class ssd_net:
    def __init__(self, config, nclasses):
        self.cfg = config
        self.nclasses = nclasses
        self.anchors, self.mapindex2nanchors = self.create_anchors()
        self.nboxes = len(self.anchors)
        self.encoded_gt_shape = (self.nboxes, 6)
        print('Number of anchor boxes: ' + str(self.nboxes))

    def get_input_shape(self):
        input_shape = [self.cfg.image_size, self.cfg.image_size]
        return input_shape

    def encode_box(self, box, anchor_idx):
        # The input box is parameterized by the top-left coordinates, and the width and height.
        # The output is the encoded box (which has offsets with respect to the center and to the width and height)
        # Relative coordinates (between 0 and 1)
        xmin = box[0]
        ymin = box[1]
        width = box[2]
        height = box[3]
        xc = xmin + width / 2.0
        yc = ymin + height / 2.0

        anchor = self.anchors[anchor_idx, :]
        xc_enc = (xc - anchor[0]) / anchor[2]  # x (center)
        yc_enc = (yc - anchor[1]) / anchor[3]  # y (center)
        w_enc = np.log(width / anchor[2])  # width
        h_enc = np.log(height / anchor[3])  # height

        # Apply 'variances':
        xc_enc /= 0.1
        yc_enc /= 0.1
        w_enc /= 0.2
        h_enc /= 0.2
        return xc_enc, yc_enc, w_enc, h_enc

    def decode_box(self, box_enc, anchor_idx):
        # The input is the encoded box (which has offsets with respect to the center and to the width and height)
        # The output box is parameterized by the top-left coordinates, and the width and height.
        # Relative coordinates (between 0 and 1)
        xc_enc = box_enc[0]
        yc_enc = box_enc[1]
        w_enc = box_enc[2]
        h_enc = box_enc[3]

        # Take into account the 'variances':
        xc_enc *= 0.1
        yc_enc *= 0.1
        w_enc *= 0.2
        h_enc *= 0.2

        anchor = self.anchors[anchor_idx, :]
        xc = xc_enc * anchor[2] + anchor[0]
        yc = yc_enc * anchor[3] + anchor[1]
        width = np.exp(w_enc) * anchor[2]
        height = np.exp(h_enc) * anchor[3]


        xmin = xc - width / 2.0
        ymin = yc - height / 2.0
        # Ensure the box fits in the image:
        xmin = min(max(xmin, 0), 1)
        ymin = min(max(ymin, 0), 1)
        width = min(max(width, 0), 1 - xmin)
        height = min(max(height, 0), 1 - ymin)
        return xmin, ymin, width, height

    def encode_gt(self, gt_boxes):
        # Inputs:
        #     gt_boxes: List of GroundTruthBox objects, with relative coordinates (between 0 and 1)
        # Outputs:
        #     encoded_labels: numpy array of dimension nboxes x 6, being nboxes the total number of default boxes
        #                     (of every feature map, every anchor, and every position). The second dimension is like
        #                     follows: First the 4 encoded coordinates of the bounding box, then a flag indicating
        #                     if it matches a ground truth box or not, and then the class id.
        #
        # We don't consider the batch dimension here.

        n_gt = len(gt_boxes)

        mask_flat = np.zeros(shape=(self.nboxes), dtype=np.bool)
        labels_flat = np.ones(shape=(self.nboxes), dtype=np.int32) * self.nclasses
        localizations_flat = np.zeros(shape=(self.nboxes, 4), dtype=np.float32)

        gt_vec = np.zeros(shape=(n_gt, 4), dtype=np.float32)
        for gt_idx in range(n_gt):
            gt_vec[gt_idx, :] = gt_boxes[gt_idx].get_coords()
        iou = compute_iou_flat(gt_vec, self.anchors) # (n_gt, nboxes)
        max_iou_per_gt = np.max(iou, axis=1)
        for gt_idx in range(iou.shape[0]):
            for anc_idx in range(iou.shape[1]):
                if np.abs(iou[gt_idx, anc_idx] - max_iou_per_gt[gt_idx]) < 1e-4:
                    iou[gt_idx, anc_idx] = max_iou_per_gt[gt_idx]
        max_iou_vec = np.max(iou, axis=1) # (n_gt)
        correspondences = -np.ones(shape=(self.nboxes), dtype=np.int32)
        if n_gt > 0:
            indices = np.argsort(-max_iou_vec)
            for gt_idx in indices:
                found_here = False
                for anc_idx in range(self.nboxes):
                    if np.abs(iou[gt_idx, anc_idx] - max_iou_vec[gt_idx]) < 1e-4:
                        if correspondences[anc_idx] < 0:
                            correspondences[anc_idx] = gt_idx
                            found_here = True
                            break
                if not found_here:
                    anchors_indices = np.argsort(-iou[gt_idx, :])
                    for anchor_idx in anchors_indices:
                        if correspondences[anchor_idx] < 0:
                            correspondences[anchor_idx] = gt_idx
                            break
            for anchor_idx in range(self.nboxes):
                if correspondences[anchor_idx] < 0:
                    gt_idx = np.argmax(iou[:, anchor_idx])
                    if iou[gt_idx, anchor_idx] > self.cfg.th_match:
                        correspondences[anchor_idx] = gt_idx
        for anchor_idx, gt_idx in enumerate(correspondences):
            if gt_idx >= 0:
                box = gt_boxes[gt_idx]
                xc_enc, yc_enc, w_enc, h_enc = self.encode_box(box.get_coords(), anchor_idx)
                mask_flat[anchor_idx] = True
                labels_flat[anchor_idx] = box.classid
                localizations_flat[anchor_idx, :] = [xc_enc, yc_enc, w_enc, h_enc]

        mask_flat = np.expand_dims(mask_flat, axis=1).astype(np.float32) # (nboxes, 1)
        labels_flat = np.expand_dims(labels_flat, axis=1).astype(np.float32) # (nboxes, 1)
        encoded_labels = np.concatenate((localizations_flat, mask_flat, labels_flat), axis=1) # shape: (nboxes, 6)

        return encoded_labels


    def decode_gt(self, encoded_labels, remove_duplicated=True):
        # Inputs:
        #     encoded_labels: numpy array of dimension nboxes x 6, being nboxes the total number of default boxes
        #                     (of every feature map, every anchor, and every position). The second dimension is like
        #                     follows: First the 4 encoded coordinates of the bounding box, then a flag indicating
        #                     if it matches a ground truth box or not, and then the class id.
        #     remove_duplicated: In SSD, many default boxes can be assigned to the same ground truth box. Therefore,
        #                        when decoding, we can find several times the same box. If we set this to True, then
        #                        the dubplicated boxes are deleted.
        # Outputs:
        #     gt_boxes: List of BoundingBox objects, with relative coordinates (between 0 and 1)
        #
        # We don't consider the batch dimension here.
        gt_boxes = []

        for anchor_idx in range(self.nboxes):
            if encoded_labels[anchor_idx, 4] > 0.5:
                xmin, ymin, width, height = self.decode_box(encoded_labels[anchor_idx, :4], anchor_idx)
                classid = int(np.round(encoded_labels[anchor_idx, 5]))
                gtbox = BoundingBox([xmin, ymin, width, height], classid)
                # Skip if it is duplicated:
                if remove_duplicated:
                    if not check_duplicated(gt_boxes, gtbox):
                        gt_boxes.append(gtbox)
                else:
                    gt_boxes.append(gtbox)
        return gt_boxes

    def decode_preds(self, encoded_preds, th_conf):
        # Inputs:
        #     encoded_preds: numpy array of dimension nboxes x (4 + nclasses + 1), being nboxes the total number of default boxes
        #                    (of every feature map, every anchor, and every position). The second dimension is like
        #                    follows: First the 4 encoded coordinates of the bounding box, and then the class id (including
        #                    a class for the background).
        # Outputs:
        #     predictions: List of PredictedBox objects, with relative coordinates (between 0 and 1)
        #
        # We don't consider the batch dimension here.
        max_positive_conf_all_boxes = np.max(encoded_preds[:, 5:], axis=1)
        sort_indices = np.argsort(-max_positive_conf_all_boxes)
        indices_keep = sort_indices[:self.cfg.max_detections_per_image]
        predictions = []
        for idx in indices_keep:
            if max_positive_conf_all_boxes[idx] > th_conf:
                xmin, ymin, width, height = self.decode_box(encoded_preds[idx, :4], idx)
                class_id = np.argmax(encoded_preds[idx, 5:])
                gtbox = PredictedBox([xmin, ymin, width, height], class_id, max_positive_conf_all_boxes[idx])
                predictions.append(gtbox)
        return predictions


    def create_anchors(self):
        anchors_list = []
        mapindex2nanchors = []
        print('Number of feature maps: ' + str(len(self.cfg.anchor_ratios)))
        for map_idx in range(len(self.cfg.anchor_ratios)):
            grid_size = self.cfg.grid_sizes[map_idx]
            box_size = self.cfg.anchor_sizes[map_idx]
            next_size = self.cfg.anchor_sizes[map_idx + 1]
            ratios = self.cfg.anchor_ratios[map_idx]
            step = self.cfg.steps[map_idx]
            grid = np.linspace(0.5 * step, (0.5 + grid_size - 1) * step, grid_size) / float(self.cfg.image_size)
            if 1 in ratios and next_size is not None:
                # Add default box with aspect ratio 1, and size sqrt(box_size * next_size)
                nanchors = len(ratios) + 1
                sizes = np.zeros(nanchors)
                sizes[:] = box_size
                sizes[1] = np.sqrt(box_size * next_size)
                ratios.insert(1, 1)
            else:
                nanchors = len(ratios)
                sizes = np.zeros(nanchors)
                sizes[:] = box_size
            mapindex2nanchors.append(nanchors)
            for anc_idx in range(len(ratios)):
                width = sizes[anc_idx] * np.sqrt(ratios[anc_idx])
                height = sizes[anc_idx] / np.sqrt(ratios[anc_idx])
                for row in range(grid_size):
                    for col in range(grid_size):
                        anchors_list.append([grid[row], grid[col], width, height])
        anchors = np.array(anchors_list)
        return anchors, mapindex2nanchors


    def build(self, inputs, cfg, is_training):
        net_output, predictions, end_points = self.net(inputs, cfg.feat_layers_names, self.cfg.grid_sizes, is_training=is_training)
        return net_output, predictions

    def net(self,
            inputs,
            feat_layers,
            grid_sizes,
            is_training=True,
            dropout_keep_prob=0.5,
            reuse=None,
            scope='vgg_16'):

        # End_points collect relevant activations for external use.
        end_points = {}
        # inputs = inputs * 255
        ch1 = inputs[:, :, :, 0]
        ch2 = inputs[:, :, :, 1]
        ch3 = inputs[:, :, :, 2]
        inputs = tf.stack([ch3, ch2, ch1], axis=3)
        with tf.variable_scope(scope, 'vgg_16', [inputs], reuse=reuse):
            # Original VGG-16 blocks.
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1', weights_initializer=tf.initializers.he_normal())  # (?, 300, 300, 64)
            end_points['conv1'] = net
            net = slim.max_pool2d(net, [2, 2], scope='pool1')  # (?, 150, 150, 64)
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2', weights_initializer=tf.initializers.he_normal())  # (?, 150, 150, 128)
            end_points['conv2'] = net
            net = slim.max_pool2d(net, [2, 2], scope='pool2')  # (?, 75, 75, 128)
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3', weights_initializer=tf.initializers.he_normal())  # (?, 75, 75, 256)
            end_points['conv3'] = net
            net = slim.max_pool2d(net, [2, 2], scope='pool3', padding='SAME')  # (?, 38, 38, 256)
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4', weights_initializer=tf.initializers.he_normal())  # (?, 38, 38, 512)
            end_points['conv4_3'] = net
            net = slim.max_pool2d(net, [2, 2], scope='pool4')  # (?, 19, 19, 512)
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5', weights_initializer=tf.initializers.he_normal())  # (?, 19, 19, 512)
            end_points['conv5'] = net
            net = slim.max_pool2d(net, [3, 3], stride=(1, 1), scope='pool5', padding='SAME')  # (?, 19, 19, 512)

            # Additional SSD blocks.
            # Block 6:
            net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='fc6_sub', weights_initializer=tf.initializers.he_normal())  # (?, 19, 19, 1024)
            end_points['fc6'] = net
            # Block 7:
            net = slim.conv2d(net, 1024, [1, 1], scope='fc7_sub', weights_initializer=tf.initializers.he_normal())  # (?, 19, 19, 1024)
            end_points['fc7'] = net
            # Block 8/9/10/11:
            end_point = 'conv6_2'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, 256, [1, 1], scope='conv1x1', weights_initializer=tf.initializers.he_normal())  # (?, 19, 19, 256)
                net = tf.pad(net, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]), "CONSTANT")
                net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID', weights_initializer=tf.initializers.he_normal())  # (?, 10, 10, 512)
            end_points[end_point] = net
            end_point = 'conv7_2'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, 128, [1, 1], scope='conv1x1', weights_initializer=tf.initializers.he_normal())  # (?, 10, 10, 128)
                net = tf.pad(net, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]), "CONSTANT")
                net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID', weights_initializer=tf.initializers.he_normal())  # (?, 5, 5, 256)
            end_points[end_point] = net
            end_point = 'conv8_2'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, 128, [1, 1], scope='conv1x1', weights_initializer=tf.initializers.he_normal())  # (?, 5, 5, 128)
                net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID', weights_initializer=tf.initializers.he_normal())  # (?, 3, 3, 256)
            end_points[end_point] = net
            end_point = 'conv9_2'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, 128, [1, 1], scope='conv1x1', weights_initializer=tf.initializers.he_normal())  # (?, 3, 3, 128)
                net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID', weights_initializer=tf.initializers.he_normal())  # (?, 1, 1, 256)
            end_points[end_point] = net

            # Prediction and localisations layers.
            batch_size = tf.shape(net)[0]
            logits = tf.zeros((batch_size, 0, self.nclasses + 1), tf.float32)
            localizations = tf.zeros((batch_size, 0, 4), tf.float32)
            for i, layer_name in enumerate(feat_layers):
                logits_this_map, locs_this_map = \
                    self.ssd_multibox_layer(end_points[layer_name], self.mapindex2nanchors[i], 'conv4' in layer_name)
                logits = tf.concat([logits, logits_this_map], axis=1)
                localizations = tf.concat([localizations, locs_this_map], axis=1)

            net_output = tf.concat([localizations, logits], axis=2)  # (batch_size, nboxes, 4 + nclasses + 1)

            return net_output, end_points


    def ssd_multibox_layer(self, inputs, nanchors, normalize):
        net = inputs  # (batch_size, grid_size, grid_size, nchannels)
        grid_size = net.shape[1]
        # Normalize:
        if normalize:
            net = l2_normalization(net)
        # Location.
        num_loc_pred = nanchors * 4
        loc_pred = slim.conv2d(net, num_loc_pred, [3, 3], activation_fn=None, scope='loc', weights_initializer=tf.initializers.he_normal())
        loc_pred = tf.reshape(loc_pred, [-1, grid_size * grid_size * nanchors, 4])
        # Class prediction.
        num_cls_pred = nanchors * (self.nclasses + 1) # We add 1 here for the background class
        cls_pred = slim.conv2d(net, num_cls_pred, [3, 3], activation_fn=None, scope='conf', weights_initializer=tf.initializers.he_normal())
        cls_pred = tf.reshape(cls_pred, [-1, grid_size * grid_size * nanchors, self.nclasses + 1])
        return cls_pred, loc_pred

    def ssdloss(self, net_output, labels, negative_ratio, args):
        # net_output: [batch_size, nboxes, 4 + nclasses + 1]
        # labels: [batch_size, nboxes, 6]

        batch_size, n_boxes, _ = labels.shape

        pred_locs = net_output[:, :, :4]
        logits = net_output[:, :, 4:]

        gt_localizations = labels[:, :, :4]  # (batch_size, nboxes, 4)
        gt_mask = labels[:, :, 4]  # (batch_size, nboxes)
        gt_labels = tf.cast(labels[:, :, 5], tf.int32)  # (batch_size, nboxes)

        background_labels = tf.equal(gt_labels, self.nclasses)
        gt_labels = gt_labels + 1
        gt_labels = tf.where(background_labels, tf.zeros(shape=(args.batch_size, self.nboxes), dtype=tf.int32),
                             gt_labels)

        prior_loss_conf = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=gt_labels)

        # Select negatives to contribute to loss:
        gt_mask_int = tf.cast(gt_mask > 0.5, tf.int32)  # (batch_size, nboxes)
        n_positives = tf.reduce_sum(gt_mask_int)  # ()
        # n_positives = tf.Print(n_positives, [n_positives], 'n_positives')

        negatives_all = 1 - gt_mask
        total_negatives = tf.cast(tf.reduce_sum(negatives_all), tf.int32)
        n_negatives_keep = tf.cast(tf.minimum(tf.maximum(1, n_positives * negative_ratio), total_negatives), tf.int32)
        neg_loss_conf_all = tf.reshape(prior_loss_conf * negatives_all, [-1])  # flatten
        _, indices = tf.nn.top_k(neg_loss_conf_all, k=n_negatives_keep, sorted=False)
        negatives_keep = tf.scatter_nd(indices=tf.expand_dims(indices, axis=1),
                                       updates=tf.ones_like(indices, dtype=tf.int32), shape=tf.shape(neg_loss_conf_all))
        negatives_keep = tf.cast(tf.reshape(negatives_keep, [-1, n_boxes]), tf.float32)

        # Localization loss:
        x = tf.subtract(gt_localizations, pred_locs)
        smooth_l1_sign = tf.cast(tf.less(tf.abs(x), 1.0), tf.float32)
        smooth_l1_option1 = tf.multiply(tf.multiply(x, x), 0.5)
        smooth_l1_option2 = tf.subtract(tf.abs(x), 0.5)
        smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                                  tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))
        smooth_l1_sum = tf.reduce_sum(smooth_l1_result, axis=2)
        # smooth_l1_sum = tf.Print(smooth_l1_sum, [tf.reduce_mean(smooth_l1_sum)], 'prior loss loc')
        loss_loc = gt_mask * smooth_l1_sum

        loss_loc = tf.reduce_sum(loss_loc)
        loss_loc = tf.cond(n_positives > 0, lambda: tf.divide(loss_loc, tf.cast(n_positives, tf.float32)),
                           lambda: tf.zeros([], tf.float32))  # why this??
        # loss_loc = tf.Print(loss_loc, [loss_loc], 'loss_loc')
        tf.summary.scalar('loss_loc', loss_loc)

        # Confidence loss:
        pos_loss_conf = tf.reduce_sum(prior_loss_conf * gt_mask)
        # pos_loss_conf = tf.Print(pos_loss_conf, [tf.reduce_sum(pos_loss_conf / tf.maximum(1.0, tf.cast(n_positives, tf.float32)))], 'pos_loss_conf')
        neg_loss_conf = tf.reduce_sum(prior_loss_conf * negatives_keep)
        # neg_loss_conf = tf.Print(neg_loss_conf, [tf.reduce_sum(neg_loss_conf / tf.maximum(1.0, tf.cast(n_positives, tf.float32)))], 'neg_loss_conf')
        loss_conf = pos_loss_conf + neg_loss_conf
        loss_conf = tf.cond(n_positives > 0, lambda: tf.divide(loss_conf, tf.cast(n_positives, tf.float32)),
                            lambda: tf.zeros([], tf.float32))  # why this??
        tf.summary.scalar('loss_conf', loss_conf)

        # Regularization is done outside.

        # Total loss:
        total_loss = loss_loc + loss_conf
        # total_loss = tf.Print(total_loss, [total_loss], 'total_loss: ')
        tf.summary.scalar('total_loss', total_loss)

        return total_loss


def compute_iou_flat(boxes, anchors):
    # box (n_gt, 4) [xmin, ymin, width, height]  # Parameterized with the top-left coordinates, and the width and height.
    # anchors (nboxes, 4)  # Parameterized with the center coordinates, and the width and height.
    # Coordinates are relative (between 0 and 1)
    nboxes = anchors.shape[0]
    n_gt = boxes.shape[0]
    boxes_expanded = np.expand_dims(boxes, axis=1) # (n_gt, 1, 4)
    boxes_expanded = np.tile(boxes_expanded, (1, nboxes, 1)) # (n_gt, nboxes, 4)
    anchors_expanded = np.expand_dims(anchors, axis=0) # (1, nboxes, 4)
    anchors_expanded = np.tile(anchors_expanded, (n_gt, 1, 1)) # (n_gt, nboxes, 4)
    xmin = np.max(np.stack((boxes_expanded[:, :, 0], anchors_expanded[:, :, 0] - anchors_expanded[:, :, 2] / 2.0), axis=2), axis=2) # (n_gt, nboxes)
    ymin = np.max(np.stack((boxes_expanded[:, :, 1], anchors_expanded[:, :, 1] - anchors_expanded[:, :, 3] / 2.0), axis=2), axis=2) # (n_gt, nboxes)
    xmax = np.min(np.stack((boxes_expanded[:, :, 0] + boxes_expanded[:, :, 2], anchors_expanded[:, :, 0] + anchors_expanded[:, :, 2] / 2.0), axis=2), axis=2) # (n_gt, nboxes)
    ymax = np.min(np.stack((boxes_expanded[:, :, 1] + boxes_expanded[:, :, 3], anchors_expanded[:, :, 1] + anchors_expanded[:, :, 3] / 2.0), axis=2), axis=2) # (n_gt, nboxes)
    zero_grid = np.zeros((n_gt, nboxes)) # (n_gt, nboxes)
    w = np.max(np.stack((xmax - xmin, zero_grid), axis=2), axis=2) # (n_gt, nboxes)
    h = np.max(np.stack((ymax - ymin, zero_grid), axis=2), axis=2) # (n_gt, nboxes)
    area_inter = w * h # (n_gt, nboxes)
    area_anchors = anchors_expanded[:, :, 2] * anchors_expanded[:, :, 3] # (n_gt, nboxes)
    area_boxes = boxes_expanded[:, :, 2] * boxes_expanded[:, :, 3] # (n_gt, nboxes)
    area_union = area_anchors + area_boxes - area_inter # (n_gt, nboxes)
    iou = area_inter / area_union # (n_gt, nboxes)
    return iou


def l2_normalization(inputs):
    inputs_shape = inputs.get_shape()
    nchannels = inputs_shape[3]
    # Normalize along spatial dimensions.
    outputs = tf.nn.l2_normalize(inputs, axis=3, epsilon=1e-12)
    # Additional scaling.
    initial_value = tf.ones(nchannels) * 20
    scale = tf.Variable(initial_value=initial_value, trainable=True, name='scale')
    outputs = tf.multiply(outputs, scale)
    return outputs
















