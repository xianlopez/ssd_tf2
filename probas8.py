from parallel_reading import AsyncParallelReader
from model import build_model, build_anchors
import time

voc_path = '/home/xian/datasets/VOC0712'
batch_size = 12
nclasses = 20
img_size = 300

model = build_model()
model.build((None, img_size, img_size, 3))
model.summary()

anchors = build_anchors(model)

reader = AsyncParallelReader(voc_path, nclasses, anchors, img_size, batch_size, 8)
# with AsyncParallelReader(voc_path, nclasses, anchors, img_size, batch_size, 8) as reader:
with reader:
    nbatches = 50
    start = time.time()
    for i in range(nbatches):
        print("i = " + str(i))
        x, y = reader.get_batch()
    end = time.time()
    lapse = end - start
    print('Total time: %.2f s (%.2f ms per batch)' % (lapse, lapse * 1000.0 / nbatches))


