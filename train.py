import numpy as np
import random
from tensorflow import keras
import os
import shutil
from datetime import datetime

random.seed(0)
np.random.seed(0)

import tensorflow as tf
from parallel_reading import AsyncParallelReader
from loss import SSDLoss
from model import build_model, build_anchors, load_vgg16_weigths
import mean_ap
from non_maximum_suppression import batch_non_maximum_suppression_fast
import tools

nclasses = 20
img_size = 300

model = build_model()
load_vgg16_weigths(model)
model.build((None, img_size, img_size, 3))
model.summary()

anchors = build_anchors(model)

loss = SSDLoss()

optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9)

model.compile(loss=loss, optimizer=optimizer)

voc_path = '/home/xian/datasets/VOC0712'
batch_size = 12
nworkers = 6

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, update_freq='batch')

nepochs = 3

@tf.function
def train_step(batch_imgs, batch_gt):
    with tf.GradientTape() as tape:
        net_output = model(batch_imgs, training=True)
        loss_value = loss(batch_gt, net_output)
        loss_value += sum(model.losses)
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_value, net_output

with AsyncParallelReader(voc_path, nclasses, anchors, img_size, batch_size, nworkers) as reader:
    for epoch in range(nepochs):
        print("\nStart epoch ", epoch + 1)
        for batch_idx in range(reader.nbatches):
            batch_imgs, batch_gt, batch_gt_raw = reader.get_batch()
            loss_value, net_output = train_step(batch_imgs, batch_gt)
            print("    batch " + str(batch_idx + 1) + "/" + str(reader.nbatches) + ", loss: %.2e" % loss_value.numpy())

            if batch_idx % 20 == 0:
                predictions_full = tools.keep_best_class(net_output.numpy())  # (batch_size, nanchors, 6)
                predictions_nms = batch_non_maximum_suppression_fast(predictions_full, nclasses)
                predictions = tools.remove_background_predictions(predictions_nms, nclasses)
                mAP = mean_ap.mean_ap_on_batch(predictions, batch_gt_raw, nclasses)
                print('mAP = %.4f' % mAP)

