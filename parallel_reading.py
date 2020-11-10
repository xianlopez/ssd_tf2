from multiprocessing import Pool
import cv2
import os
import random
import numpy as np
from data_augmentation import data_augmentation
import encoding
from multiprocessing import Pool, RawArray, Process, Pipe
from lxml import etree

voc_classes_map = {
    'person': 0,
    'bird': 1,
    'cat': 2,
    'cow': 3,
    'dog': 4,
    'horse': 5,
    'sheep': 6,
    'aeroplane': 7,
    'bicycle': 8,
    'boat': 9,
    'bus': 10,
    'car': 11,
    'motorbike': 12,
    'train': 13,
    'bottle': 14,
    'chair': 15,
    'diningtable': 16,
    'pottedplant': 17,
    'sofa': 18,
    'tvmonitor': 19
}

max_gt_boxes = 100

image_means = np.array([123.0, 117.0, 104.0])
image_means /= 255.0
image_means = np.reshape(image_means, [1, 1, 3])

var_dict = {}


class ImageReference:
    def __init__(self, year, name):
        self.year = year
        self.name = name


def get_image_references_train(voc_path):
    trainval2007 = os.path.join(voc_path, 'VOC2007', 'ImageSets', 'Main', 'trainval.txt')
    trainval2012 = os.path.join(voc_path, 'VOC2012', 'ImageSets', 'Main', 'trainval.txt')
    assert os.path.isfile(trainval2007)
    assert os.path.isfile(trainval2012)
    references_list = []
    with open(trainval2007, 'r') as fid:
        names = fid.read().splitlines()
        for name in names:
            references_list.append(ImageReference('07', name))
    with open(trainval2012, 'r') as fid:
        names = fid.read().splitlines()
        for name in names:
            references_list.append(ImageReference('12', name))
    return references_list


def get_image_references_val(voc_path):
    test2007 = os.path.join(voc_path, 'VOC2007', 'ImageSets', 'Main', 'test.txt')
    assert os.path.isfile(test2007)
    references_list = []
    with open(test2007, 'r') as fid:
        names = fid.read().splitlines()
        for name in names:
            references_list.append(ImageReference('07', name))
    return references_list


def parse_voc_annotation(gt_path, img_height, img_width):
    # Note: img_height and img_width could be read from the annotation itself, but I think
    # it's easier to use the values read from the image itself, since I already have them.
    tree = etree.parse(gt_path)
    annotation = tree.getroot()
    objects = annotation.findall('object')
    boxes = []
    for obj in objects:
        class_name = obj.find('name').text
        class_id = voc_classes_map[class_name]
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        width = xmax - xmin + 1
        height = ymax - ymin + 1
        # Make relative:
        xmin = float(xmin) / float(img_width)
        ymin = float(ymin) / float(img_height)
        width = float(width) / float(img_width)
        height = float(height) / float(img_height)
        boxes.append([xmin, ymin, width, height, class_id])
    boxes = np.array(boxes, dtype=np.float32)
    return boxes


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
    year = inputs[2]
    position_in_batch = inputs[3]
    # Read image:
    img_path = os.path.join(opts.voc_path, 'VOC20' + year, 'JPEGImages', rawname + '.jpg')
    assert os.path.isfile(img_path)  # TODO: Remove me.
    img = cv2.imread(img_path)

    img_height, img_width, _ = img.shape

    img = img.astype(np.float32) / 255.0

    # Read ground truth:
    gt_path = os.path.join(opts.voc_path, 'VOC20' + year, 'Annotations', rawname + '.xml')
    assert os.path.isfile(gt_path)  # TODO: Remove me.
    boxes = parse_voc_annotation(gt_path, img_height, img_width)
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
    def __init__(self, opts, split):
        self.opts = opts
        self.batch_index = 0
        if split == 'train':
            self.image_references = get_image_references_train(opts.voc_path)
        elif split == 'val':
            self.image_references = get_image_references_val(opts.voc_path)
        else:
            raise Exception('Unexpected split')
        print(str(len(self.image_references)) + ' images for ' + split)
        self.nbatches = len(self.image_references) // int(self.opts.batch_size)
        # Randomize the image pairs:
        random.shuffle(self.image_references)
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
        names = []
        input_data = []
        for position_in_batch in range(self.opts.batch_size):
            data_idx = self.batch_index * self.opts.batch_size + position_in_batch
            input_data.append((self.opts, self.image_references[data_idx].name,
                               self.image_references[data_idx].year, position_in_batch))
            names.append(self.image_references[data_idx].year + '_' + self.image_references[data_idx].name)

        self.pool.map(read_image, input_data)

        batch_imgs_np = np.frombuffer(self.batch_imgs_Arr).reshape(self.batch_imgs_shape)
        batch_gt_np = np.frombuffer(self.batch_gt_Arr).reshape(self.batch_gt_shape)
        batch_gt_raw_np = np.frombuffer(self.batch_gt_raw_Arr).reshape(self.batch_gt_raw_shape)

        self.batch_index += 1
        if self.batch_index == self.nbatches:
            self.batch_index = 0
            random.shuffle(self.image_references)
            # print('Rewinding data!')

        return batch_imgs_np, batch_gt_np, batch_gt_raw_np, names


def async_reader_loop(opts, split, conn):
    print('async_reader_loop is alive!')
    reader = ParallelReader(opts, split)
    conn.send(reader.nbatches)
    batch_imgs, batch_gt, batch_gt_raw, names = reader.fetch_batch()
    while conn.recv() == 'GET':
        conn.send([batch_imgs, batch_gt, batch_gt_raw, names])
        batch_imgs, batch_gt, batch_gt_raw, names = reader.fetch_batch()
    print('async_reader_loop says goodbye!')


class AsyncParallelReader:
    def __init__(self, voc_path, nclasses, anchors, img_size, batch_size, nworkers, split):
        print('Starting AsyncParallelReader')
        opts = ReaderOpts(voc_path, nclasses, anchors, img_size, batch_size, nworkers)
        self.conn1, conn2 = Pipe()
        self.reader_process = Process(target=async_reader_loop, args=(opts, split, conn2))
        self.reader_process.start()
        self.nbatches = self.conn1.recv()

    def get_batch(self):
        self.conn1.send('GET')
        batch_imgs, batch_gt, batch_gt_raw, names = self.conn1.recv()
        return batch_imgs, batch_gt, batch_gt_raw, names

    def __exit__(self, type, value, traceback):
        print('Ending AsyncParallelReader')
        self.conn1.send('END')
        self.reader_process.join()

    def __enter__(self):
        return self

