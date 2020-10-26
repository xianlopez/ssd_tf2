import os
import random
import cv2
import encoding
import numpy as np
from data_augmentation import data_augmentation
import tensorflow
from tensorflow.keras.utils import Sequence

# TODO: I should use the original VOC dataset, not my modified version.

image_means = np.array([123.0, 117.0, 104.0])
image_means /= 255.0
image_means = np.reshape(image_means, [1, 1, 3])


class DataReader(Sequence):
    def __init__(self, voc_path, nclasses, anchors, img_size, batch_size):
        self.voc_path = voc_path
        self.nclasses = nclasses
        self.anchors = anchors
        self.img_size = img_size
        self.batch_size = batch_size
        with open(os.path.join(voc_path, 'train_files.txt'), 'r') as fid1:
            self.raw_names = [line for line in fid1.read().splitlines() if line != '']
        print(str(len(self.raw_names)) + ' images for training.')
        # Randomize the image pairs:
        random.shuffle(self.raw_names)

    def __len__(self):
        return len(self.raw_names) // int(self.batch_size)

    def __getitem__(self, idx):
        batch_imgs = np.zeros(shape=(self.batch_size, self.img_size, self.img_size, 3), dtype=np.float32)
        batch_gt = np.zeros(shape=(self.batch_size, len(self.anchors), 4 + self.nclasses + 1), dtype=np.float32)
        for i in range(self.batch_size):
            # Read image:
            img = cv2.imread(os.path.join(self.voc_path, 'images', self.raw_names[idx * self.batch_size + i] + '.jpg'))
            img = img.astype(np.float32) / 255.0

            # Read ground truth:
            boxes = []
            with open(os.path.join(self.voc_path, 'annotations', self.raw_names[idx * self.batch_size + i] + '.txt'), 'r') as fid2:
                ann_lines = [line for line in fid2.read().splitlines() if line != '']
                for line in ann_lines:
                    line_split = line.split(' ')
                    assert len(line_split) == 5
                    classid = int(line_split[0])
                    xmin = int(line_split[1])
                    ymin = int(line_split[2])
                    width = int(line_split[3])
                    height = int(line_split[4])
                    boxes.append([xmin, ymin, width, height, classid])
            boxes = np.array(boxes, dtype=np.float32)

            # Data augmentation:
            img, boxes = data_augmentation(img, boxes)

            # Preprocess image:
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img - image_means

            batch_imgs[i, :, :, :] = img
            batch_gt[i, :, :] = encoding.encode_gt(boxes, self.anchors, self.nclasses)

        return batch_imgs, batch_gt

    def on_epoch_end(self):
        print("Rewinding data.")
        random.shuffle(self.raw_names)
