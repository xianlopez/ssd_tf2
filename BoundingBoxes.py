import numpy as np


def check_duplicated(all_boxes, new_box, tolerance=1e-6):
    for old_box in all_boxes:
        if old_box.classid == new_box.classid:
            old_coords = np.array(old_box.get_coords())
            new_coords = np.array(new_box.get_coords())
            if np.sum(np.square(old_coords - new_coords)) < tolerance:
                return True
    return False


class BoundingBox:
    def __init__(self):
        self.xmin = -1
        self.ymin = -1
        self.width = -1
        self.height = -1
        self.classid = -1

    def __init__(self, coords, cl_id):
        self.xmin = coords[0]
        self.ymin = coords[1]
        self.width = coords[2]
        self.height = coords[3]
        self.classid = cl_id

    def apply_padding(self, pad_rel):
        self.xmin = pad_rel + self.xmin * (1 - 2 * pad_rel)
        self.ymin = pad_rel + self.ymin * (1 - 2 * pad_rel)
        self.width = self.width * (1 - 2 * pad_rel)
        self.height = self.height * (1 - 2 * pad_rel)

    def get_coords(self):
        return [self.xmin, self.ymin, self.width, self.height]

    def is_relative(self):
        coords = np.array(self.get_coords())
        if np.all(coords >= np.zeros(4)) and np.all(coords <= np.ones(4)):
            return True
        else:
            return False

    def convert_to_relative(self, img_width, img_height):
        if self.is_relative():
            raise Exception('Error converting to relative: image is already relative.')
        self.xmin = self.xmin / float(img_width)
        self.ymin = self.ymin / float(img_height)
        self.width = self.width / float(img_width)
        self.height = self.height / float(img_height)

    def get_abs_coords(self, img_width, img_height):
        # if not self.is_relative():
        #     self.print()
        #     raise Exception('Error converting to absolute: image is already absolute.')
        xmin = int(np.round(self.xmin * img_width))
        ymin = int(np.round(self.ymin * img_height))
        width = int(np.round(self.width * img_width))
        height = int(np.round(self.height * img_height))
        # Make sure it fits inside the image:
        xmin = min(max(xmin, 0), img_width - 1)
        ymin = min(max(ymin, 0), img_height - 1)
        width = min(max(width, 1), img_width - xmin)
        height = min(max(height, 1), img_height - ymin)
        return [xmin, ymin, width, height]

    def get_abs_coords_cv(self, cv_image):
        height, width, _ = cv_image.shape
        return self.get_abs_coords(width, height)

    def print(self):
        print(str(self.xmin) + ' ' + str(self.ymin) + ' ' + str(self.width) + ' ' + str(self.height) + ' - ' + str(self.classid))


class PredictedBox(BoundingBox):
    def __init__(self, coords, cl_id, conf):
        super(PredictedBox, self).__init__(coords, cl_id)
        self.confidence = conf
        self.tp = 'unknown'

    def set_tp(self, is_tp):
        if is_tp:
            self.tp = 'yes'
        else:
            self.tp = 'no'


def boxes_are_equal(box1, box2, tolerance):
    if box1.classid == box2.classid:
        box1_array = np.array(box1.get_coords())
        box2_array = np.array(box2.get_coords())
        return np.sum(box1_array - box2_array) < tolerance
    else:
        return False

