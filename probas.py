import tensorflow as tf
# from data_reader import DataReader
from loss import SSDLoss
from model import SSD

import numpy as np

nclasses = 20
img_size = 300

model = SSD(img_size)
model.build((None, img_size, img_size, 3))
model.summary()

loss = SSDLoss()

optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9)

model.compile(loss=loss, optimizer=optimizer)


x = np.zeros((16, img_size, img_size, 3), np.float32)
y = model(x)





steps = [8, 16, 32, 64, 100, 300]
grid_sizes = [38, 19, 10, 5, 3, 1]
img_size = 300
for i in range(len(steps)):
    grid = np.linspace(0.5 * steps[i], (0.5 + grid_sizes[i] - 1) * steps[i], grid_sizes[i]) / float(img_size)
    print(grid.shape)



step = 64
grid_size = 5
img_size = 300
grid = np.linspace(0.5 * step, (0.5 + grid_size - 1) * step, grid_size) / float(img_size)
print(grid.shape)