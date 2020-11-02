import numpy as np
from non_maximum_suppression import non_maximum_suppression_slow, non_maximum_suppression_fast
import time

nclasses = 20
nboxes = 5000
x0 = np.random.uniform(0.0, 0.6, nboxes)
y0 = np.random.uniform(0.0, 0.6, nboxes)
w = np.random.uniform(0.0, 0.4, nboxes)
h = np.random.uniform(0.0, 0.4, nboxes)
class_id = np.random.randint(0, nclasses + 1, nboxes)
conf = np.random.uniform(0.0, 1.0, nboxes)
boxes = np.stack([x0, y0, w, h, class_id, conf], axis=1)

start = time.time()
boxes_out = non_maximum_suppression_fast(boxes, nclasses)
end = time.time()
print('Fast time: %.2f ms' % ((end - start) * 1e3))


boxes_inout = boxes.copy()
start = time.time()
non_maximum_suppression_slow(boxes_inout, nclasses)
end = time.time()
print('Slow time: %.2f ms' % ((end - start) * 1e3))

