import tensorflow as tf

negative_ratio = 3


class SSDLoss(tf.losses.Loss):
    def __init__(self):
        super(SSDLoss, self).__init__(reduction='none')
        self.huber_loss = tf.losses.Huber(delta=1.0, reduction='none')

    def call(self, y_true, y_pred):
        # Input format (both y_true and y_pred):
        # (?, nanchors, 4 + nclasses + 1)
        # [xmin_enc, ymin_enc, width_enc, height_enc, class1, ..., classN, background]

        pred_locs = y_pred[:, :, :4]
        logits = y_pred[:, :, 4:]

        gt_locs = y_true[:, :, :4]
        labels = y_true[:, :, 4:]

        prior_loss_conf = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)  # (?, nanchors)

        positives_mask = labels[:, :, -1] < 0.5  # (?, nanchors)
        positives_mask_int = tf.cast(positives_mask, tf.int32)  # (?, nanchors)
        positives_mask_float = tf.cast(positives_mask_int, tf.float32)  # (?, nanchors)
        negatives_mask_float = 1 - positives_mask_float  # (?, nanchors)
        n_positives = tf.reduce_sum(positives_mask_int)  # ()
        batch_size = tf.shape(positives_mask)[0]
        nanchors = tf.shape(positives_mask)[1]
        nanchors_total = batch_size * nanchors
        n_negatives = nanchors_total - n_positives
        n_negatives_keep = tf.cast(tf.minimum(tf.maximum(1, n_positives * negative_ratio), n_negatives), tf.int32)
        neg_loss_conf_all = tf.reshape(prior_loss_conf * negatives_mask_float, [-1])  # (nanchors_total)
        _, indices = tf.nn.top_k(neg_loss_conf_all, k=n_negatives_keep, sorted=False)
        negatives_keep = tf.scatter_nd(indices=tf.expand_dims(indices, axis=1),
                                       updates=tf.ones_like(indices, dtype=tf.int32), shape=tf.shape(neg_loss_conf_all))
        negatives_keep = tf.cast(tf.reshape(negatives_keep, [-1, nanchors]), tf.float32)

        # Localization loss:
        loss_loc = positives_mask_float * self.huber_loss(gt_locs, pred_locs)  # (?, nanchors)
        loss_loc = tf.reduce_sum(loss_loc)
        loss_loc = tf.cond(n_positives > 0, lambda: tf.divide(loss_loc, tf.cast(n_positives, tf.float32)),
                           lambda: tf.zeros([], tf.float32))
        # tf.summary.scalar('loss_loc', loss_loc)

        # Confidence loss:
        pos_loss_conf = tf.reduce_sum(prior_loss_conf * positives_mask_float)
        neg_loss_conf = tf.reduce_sum(prior_loss_conf * negatives_keep)
        loss_conf = pos_loss_conf + neg_loss_conf
        loss_conf = tf.cond(n_positives > 0, lambda: tf.divide(loss_conf, tf.cast(n_positives, tf.float32)),
                            lambda: tf.zeros([], tf.float32))
        # tf.summary.scalar('loss_conf', loss_conf)

        # Regularization is done outside.

        # Total loss:
        total_loss = loss_loc + loss_conf
        # tf.summary.scalar('total_loss', total_loss)

        return total_loss

