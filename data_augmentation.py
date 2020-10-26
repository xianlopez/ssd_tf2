import numpy as np
import random
import cv2
from compute_iou import compute_iou_single


def data_augmentation(img, boxes):
    # img: (height, width, 3)
    # boxes: (nboxes, 5) [xmin, ymin, width, height, classid]

    assert np.all(img >= 0.0) and np.all(img <= 1.0)
    assert len(boxes.shape) == 2
    assert boxes.shape[1] == 5

    img = photometric_distortions(img)
    img, boxes = expand_and_crop(img, boxes)
    img, boxes = random_flip(img, boxes)

    return img, boxes


def photometric_distortions(img):
    if random.random() < 0.5:  # 50%
        return photometric_sequence_1(img)
    else:  # 50%
        return photometric_sequence_2(img)


def photometric_sequence_1(rgb):
    rgb = random_adjust_brightness(rgb)
    rgb = random_adjust_contrast(rgb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hsv = random_adjust_saturation(hsv)
    hsv = random_adjust_hue(hsv)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def photometric_sequence_2(rgb):
    rgb = random_adjust_brightness(rgb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hsv = random_adjust_saturation(hsv)
    hsv = random_adjust_hue(hsv)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    rgb = random_adjust_contrast(rgb)
    return rgb


def random_adjust_brightness(rgb):
    max_delta = 0.125  # 32 / 255.0
    delta = random.uniform(-max_delta, max_delta)
    rgb = np.clip(rgb + delta, 0.0, 1.0)
    return rgb


def random_adjust_contrast(rgb):
    factor_min = 0.5
    factor_max = 1.5
    factor = random.uniform(factor_min, factor_max)
    rgb = (rgb - 127.5) * factor + 127.5
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


def random_adjust_saturation(hsv):
    # The saturation channels is assumed to be in the interval [0, 1]
    factor_min = 0.5
    factor_max = 1.5
    factor = random.uniform(factor_min, factor_max)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0.0, 1.0)
    return hsv


def random_adjust_hue(hsv):
    max_delta = 0.1
    delta = random.uniform(-max_delta, max_delta)
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] + delta, 0.0, 1.0)
    return hsv


def expand_and_crop(img, boxes):
    # Expand 50% of the times
    img_height = img.shape[0]
    img_width = img.shape[1]
    if random.random() < 0.5:
        scale = np.random.rand() * 3.0 + 1  # between 1 and 4
        new_width = np.round(img_width * scale).astype(np.int32)
        new_height = np.round(img_height * scale).astype(np.int32)
        canvas = np.mean(img, axis=(0, 1), keepdims=True)  # (1, 1, 3)
        canvas = np.tile(canvas, [new_height, new_width, 1])  # (new_height, new_width, 3)
        pos_i = np.random.randint(new_height - img_height + 1)
        pos_j = np.random.randint(new_width - img_width + 1)
        canvas[pos_i:(pos_i+img_height), pos_j:(pos_j+img_width), :] = img
        img = canvas
        boxes[:, 0] = (pos_j + boxes[:, 0] * img_width) / float(new_width)
        boxes[:, 1] = (pos_i + boxes[:, 1] * img_height) / float(new_height)
        boxes[:, 2] = boxes[:, 2] / scale
        boxes[:, 3] = boxes[:, 3] / scale
    else:
        new_width = img_width
        new_height = img_height

    # Random crop:
    min_scale = 0.3
    max_scale = 1
    min_aspect_ratio = 0.5
    max_aspect_ratio = 2
    iou_th_list = [-1, 0.1, 0.3, 0.5, 0.7, 0.9]
    while True:  # Keep going until we either find a valid patch or return the original image.
        if np.random.rand() >= (1 - 0.857):
            iou_th = iou_th_list[np.random.randint(len(iou_th_list))]
            for _ in range(50):
                # TODO: I think I can get rid of the "rel" conversion here, and directly work with absolute coords all the time
                patch_x0_rel, patch_y0_rel, patch_width_rel, patch_height_rel = \
                    make_patch_shape(new_width, new_height, min_scale, max_scale, min_aspect_ratio, max_aspect_ratio)
                patch_x1_rel = patch_x0_rel + patch_width_rel
                patch_y1_rel = patch_y0_rel + patch_height_rel
                # Check boxes' IOU:
                patch_is_valid = False
                for i in range(boxes.shape[0]):
                    x_center = boxes[i, 0] + float(boxes[i, 2]) / 2
                    y_center = boxes[i, 1] + float(boxes[i, 3]) / 2
                    if patch_x0_rel < x_center < patch_x1_rel and patch_y0_rel < y_center < patch_y1_rel:
                        if compute_iou_single([patch_x0_rel, patch_y0_rel, patch_width_rel, patch_height_rel], boxes[i, :4]) > iou_th:
                            patch_is_valid = True
                            break
                if patch_is_valid:
                    img, boxes = sample_patch(img, boxes, new_width, new_height, patch_x0_rel, patch_y0_rel, patch_width_rel, patch_height_rel)
                    return img, boxes
        else:
            return img, boxes


def make_patch_shape(img_width, img_height, min_scale, max_scale, min_aspect_ratio, max_aspect_ratio):
    scale = np.random.rand() * (max_scale - min_scale) + min_scale
    aspect_ratio = np.random.rand() * (max_aspect_ratio - min_aspect_ratio) + min_aspect_ratio
    patch_width = np.sqrt(aspect_ratio * scale * float(img_width) * float(img_height))
    patch_height = np.sqrt(scale * float(img_width) * float(img_height) / aspect_ratio)
    patch_width = np.minimum(np.maximum(np.round(patch_width), 1), img_width).astype(np.int32)
    patch_height = np.minimum(np.maximum(np.round(patch_height), 1), img_height).astype(np.int32)
    x0 = np.random.randint(img_width - patch_width + 1)
    y0 = np.random.randint(img_height - patch_height + 1)
    # Convert to relative coordinates:
    x0 = x0 / float(img_width)
    y0 = y0 / float(img_height)
    patch_width = patch_width / float(img_width)
    patch_height = patch_height / float(img_height)
    return x0, y0, patch_width, patch_height


def sample_patch(img, boxes, img_width, img_height, patch_x0_rel, patch_y0_rel, patch_width_rel, patch_height_rel):
    # Convert to absolute coordinates:
    patch_x0_abs = max(np.round(patch_x0_rel * img_width).astype(np.int32), 0)
    patch_y0_abs = max(np.round(patch_y0_rel * img_height).astype(np.int32), 0)
    patch_width_abs = min(np.round(patch_width_rel * img_width).astype(np.int32), img_width - patch_x0_abs)
    patch_height_abs = min(np.round(patch_height_rel * img_height).astype(np.int32), img_height - patch_y0_abs)
    # Image:
    patch = img[patch_y0_abs:(patch_y0_abs+patch_height_abs), patch_x0_abs:(patch_x0_abs+patch_width_abs), :]
    # Bounding boxes:
    patch_x1_rel = patch_x0_rel + patch_width_rel
    patch_y1_rel = patch_y0_rel + patch_height_rel
    remaining_boxes_list = []
    for i in range(boxes.shape[0]):
        x_center = boxes[i, 0] + float(boxes[i, 2]) / 2
        y_center = boxes[i, 1] + float(boxes[i, 3]) / 2
        if patch_x0_rel < x_center < patch_x1_rel and patch_y0_rel < y_center < patch_y1_rel:
            new_box_x0 = max((boxes[i, 0] - patch_x0_rel) / patch_width_rel, 0.0)
            new_box_y0 = max((boxes[i, 1] - patch_y0_rel) / patch_height_rel, 0.0)
            new_box_x1 = min((boxes[i, 0] + boxes[i, 2] - patch_x0_rel) / patch_width_rel, 1.0)
            new_box_y1 = min((boxes[i, 1] + boxes[i, 3] - patch_y0_rel) / patch_height_rel, 1.0)
            new_box_width = new_box_x1 - new_box_x0
            new_box_height = new_box_y1 - new_box_y0
            remaining_boxes_list.append([new_box_x0, new_box_y0, new_box_width, new_box_height, boxes[i, 4]])
    remaining_boxes = np.array(remaining_boxes_list, dtype=np.float32)
    return patch, remaining_boxes


def random_flip(img, boxes):
    img = cv2.flip(img, 1)
    boxes[:, 0] = 1.0 - boxes[:, 0] - boxes[:, 2]
    return img, boxes
