import numpy as np
import random

random.seed(0)
np.random.seed(0)

import tensorflow as tf
from data_reader import DataReader
from loss import SSDLoss
from model import build_model, build_anchors

nclasses = 20
img_size = 300

model = build_model()
model.build((None, img_size, img_size, 3))
model.summary()

anchors = build_anchors(model)

loss = SSDLoss()

optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9)

model.compile(loss=loss, optimizer=optimizer)

voc_path = '/home/xian/datasets/VOC0712'
batch_size = 8
reader = DataReader(voc_path, nclasses, anchors, img_size, batch_size)

model.fit(reader, epochs=3)





