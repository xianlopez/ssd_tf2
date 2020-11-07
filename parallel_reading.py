from multiprocessing import Pool
import cv2
import os
import random
import numpy as np
from data_augmentation import data_augmentation
import encoding
from multiprocessing import Pool, RawArray, Process, Pipe

# TODO: I should use the original VOC dataset, not my modified version.

max_gt_boxes = 100

image_means = np.array([123.0, 117.0, 104.0])
image_means /= 255.0
image_means = np.reshape(image_means, [1, 1, 3])

var_dict = {}


def init_worker(batch_imgs_Arr, batch_imgs_shape, batch_gt_Arr, batch_gt_shape, batch_gt_raw_Arr, batch_gt_raw_shape):
    var_dict['batch_imgs_Arr'] = batch_imgs_Arr
    var_dict['batch_imgs_shape'] = batch_imgs_shape
    var_dict['batch_gt_Arr'] = batch_gt_Arr
    var_dict['batch_gt_shape'] = batch_gt_shape
    var_dict['batch_gt_raw_Arr'] = batch_gt_raw_Arr
    var_dict['batch_gt_raw_shape'] = batch_gt_raw_shape


def read_image(inputs):
    opts = inputs[0]
    rawname = inputs[1]
    position_in_batch = inputs[2]
    # Read image:
    img = cv2.imread(os.path.join(opts.voc_path, 'images', rawname + '.jpg'))

    img_height, img_width, _ = img.shape

    img = img.astype(np.float32) / 255.0

    # Read ground truth:
    boxes = []
    with open(os.path.join(opts.voc_path, 'annotations', rawname + '.txt'), 'r') as fid2:
        ann_lines = [line for line in fid2.read().splitlines() if line != '']
        for line in ann_lines:
            line_split = line.split(' ')
            assert len(line_split) == 5
            classid = int(line_split[0])
            xmin = int(line_split[1])
            ymin = int(line_split[2])
            width = int(line_split[3])
            height = int(line_split[4])
            # Make relative:
            xmin = float(xmin) / float(img_width)
            ymin = float(ymin) / float(img_height)
            width = float(width) / float(img_width)
            height = float(height) / float(img_height)
            boxes.append([xmin, ymin, width, height, classid])
    boxes = np.array(boxes, dtype=np.float32)
    assert boxes.shape[0] <= max_gt_boxes

    # Data augmentation:
    img, boxes = data_augmentation(img, boxes)

    # Preprocess image:
    img = cv2.resize(img, (opts.img_size, opts.img_size))
    img = img - image_means

    # Expand raw boxes:
    boxes_raw = np.zeros((max_gt_boxes, 5), np.float32)
    boxes_raw[:boxes.shape[0], :] = boxes
    boxes_raw[boxes.shape[0]:, -1] = opts.nclasses  # Mark the rest as background

    # Wrap shared data as numpy arrays:
    batch_imgs_np = np.frombuffer(var_dict['batch_imgs_Arr']).reshape(var_dict['batch_imgs_shape'])
    batch_gt_np = np.frombuffer(var_dict['batch_gt_Arr']).reshape(var_dict['batch_gt_shape'])
    batch_gt_raw_np = np.frombuffer(var_dict['batch_gt_raw_Arr']).reshape(var_dict['batch_gt_raw_shape'])

    # Assign values:
    batch_imgs_np[position_in_batch, :, :, :] = img
    batch_gt_np[position_in_batch, :, :] = encoding.encode_gt(boxes, opts.anchors, opts.nclasses)
    batch_gt_raw_np[position_in_batch, :, :] = boxes_raw

    return


class ReaderOpts:
    def __init__(self, voc_path, nclasses, anchors, img_size, batch_size, nworkers):
        self.voc_path = voc_path
        self.nclasses = nclasses
        self.anchors = anchors
        self.img_size = img_size
        self.batch_size = batch_size
        self.nworkers = nworkers


class ParallelReader:
    def __init__(self, opts):
        self.opts = opts
        self.batch_index = 0
        with open(os.path.join(self.opts.voc_path, 'train_files.txt'), 'r') as fid1:
            self.raw_names = [line for line in fid1.read().splitlines() if line != '']
        print(str(len(self.raw_names)) + ' images for training.')
        self.nbatches = len(self.raw_names) // int(self.opts.batch_size)
        # Randomize the image pairs:
        random.shuffle(self.raw_names)
        # Initialize batch buffers:
        self.batch_imgs_shape = (self.opts.batch_size, self.opts.img_size, self.opts.img_size, 3)
        self.batch_gt_shape = (self.opts.batch_size, len(self.opts.anchors), 4 + self.opts.nclasses + 1)
        self.batch_gt_raw_shape = (self.opts.batch_size, max_gt_boxes, 5)
        self.batch_imgs_Arr = RawArray('d', self.batch_imgs_shape[0] * self.batch_imgs_shape[1] *
                                  self.batch_imgs_shape[2] * self.batch_imgs_shape[3])
        self.batch_gt_Arr = RawArray('d', self.batch_gt_shape[0] * self.batch_gt_shape[1] * self.batch_gt_shape[2])
        self.batch_gt_raw_Arr = RawArray('d', self.batch_gt_raw_shape[0] * self.batch_gt_raw_shape[1] *
                                         self.batch_gt_raw_shape[2])
        # Initialize pool:
        self.pool = Pool(processes=self.opts.nworkers, initializer=init_worker, initargs=
            (self.batch_imgs_Arr, self.batch_imgs_shape, self.batch_gt_Arr, self.batch_gt_shape,
             self.batch_gt_raw_Arr, self.batch_gt_raw_shape))

    def fetch_batch(self):
        input_data = []
        for position_in_batch in range(self.opts.batch_size):
            data_idx = self.batch_index * self.opts.batch_size + position_in_batch
            input_data.append((self.opts, self.raw_names[data_idx], position_in_batch))

        self.pool.map(read_image, input_data)

        batch_imgs_np = np.frombuffer(self.batch_imgs_Arr).reshape(self.batch_imgs_shape)
        batch_gt_np = np.frombuffer(self.batch_gt_Arr).reshape(self.batch_gt_shape)
        batch_gt_raw_np = np.frombuffer(self.batch_gt_raw_Arr).reshape(self.batch_gt_raw_shape)

        self.batch_index += 1
        if self.batch_index == self.nbatches:
            self.batch_index = 0
            random.shuffle(self.raw_names)
            print('Rewinding data!')

        return batch_imgs_np, batch_gt_np, batch_gt_raw_np


def async_reader_loop(opts, conn):
    print('async_reader_loop is alive!')
    reader = ParallelReader(opts)
    conn.send(reader.nbatches)
    batch_imgs, batch_gt, batch_gt_raw = reader.fetch_batch()
    while conn.recv() == 'GET':
        conn.send([batch_imgs, batch_gt, batch_gt_raw])
        batch_imgs, batch_gt, batch_gt_raw = reader.fetch_batch()
    print('async_reader_loop says goodbye!')


class AsyncParallelReader:
    def __init__(self, voc_path, nclasses, anchors, img_size, batch_size, nworkers):
        print('Starting AsyncParallelReader')
        opts = ReaderOpts(voc_path, nclasses, anchors, img_size, batch_size, nworkers)
        self.conn1, conn2 = Pipe()
        self.reader_process = Process(target=async_reader_loop, args=(opts, conn2))
        self.reader_process.start()
        self.nbatches = self.conn1.recv()

    def get_batch(self):
        self.conn1.send('GET')
        batch_imgs, batch_gt, batch_gt_raw = self.conn1.recv()
        return batch_imgs, batch_gt, batch_gt_raw

    def __exit__(self, type, value, traceback):
        print('Ending AsyncParallelReader')
        self.conn1.send('END')
        self.reader_process.join()

    def __enter__(self):
        return self

