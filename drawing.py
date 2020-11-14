import cv2
import numpy as np
from decoding import decode_preds, decode_gt_batch
from parallel_reading import image_means


def draw_boxes(img, boxes):
    # img: (height, width, 3)
    # boxes: list of boxes. Each box comes as [xmin, ymin, width, height].
    height, width, _ = img.shape
    color = (1.0, 0.0, 0.0)
    for i in range(len(boxes)):
        box_xmin_abs = min(max(int(round(boxes[i][0] * width)), 0), width - 2)
        box_ymin_abs = min(max(int(round(boxes[i][1] * height)), 0), height - 2)
        box_width_abs = max(int(round(boxes[i][2] * width)), 1)
        box_height_abs = max(int(round(boxes[i][3] * height)), 1)
        box_xmax_abs = min(box_xmin_abs + box_width_abs, width - 1)
        box_ymax_abs = min(box_ymin_abs + box_height_abs, height - 1)
        img = cv2.rectangle(img, (box_xmin_abs, box_ymin_abs), (box_xmax_abs, box_ymax_abs), color, thickness=2)
    return img


def display_anchors(img, anchors):
    # anchors: (nanchors, 4) [xmin, ymin, width, height]
    height, width, _ = img.shape
    color = (0.0, 1.0, 1.0)
    # for i in range(2000, anchors.shape[0]):
    for i in range(anchors.shape[0]):
        img_to_show = img.copy()
        anc_xmin_abs = int(round(anchors[i, 0] * width))
        anc_ymin_abs = int(round(anchors[i, 1] * height))
        anc_width_abs = int(round(anchors[i, 2] * width))
        anc_height_abs = int(round(anchors[i, 3] * height))
        anc_xmax_abs = anc_xmin_abs + anc_width_abs
        anc_ymax_abs = anc_ymin_abs + anc_height_abs
        img_to_show = cv2.rectangle(img_to_show, (anc_xmin_abs, anc_ymin_abs),
                                    (anc_xmax_abs, anc_ymax_abs), color, thickness=2)
        cv2.imshow('anchors', img_to_show)
        cv2.waitKey(100)


def display_gt_and_preds(net_output, batch_imgs, batch_gt_raw, batch_gt, anchors, nclasses):
    predictions = decode_preds(net_output.numpy(), anchors, nclasses)
    img = batch_imgs[0, ...]
    img = img + image_means
    img_gt = img.copy()
    gt = np.take(batch_gt_raw[0], np.where(batch_gt_raw[0][:, 4] != nclasses)[0], axis=0)  # (ngt, 5)
    img_gt = draw_boxes(img_gt, gt)
    cv2.imshow('image with GT', img_gt)
    img_gt_dec = img.copy()
    gt_dec = decode_gt_batch(batch_gt, anchors)[0]  # (ngt_dec, 5)
    img_gt_dec = draw_boxes(img_gt_dec, gt_dec)
    cv2.imshow('image with decoded GT', img_gt_dec)
    preds_img = predictions[0]
    img_with_boxes = img.copy()
    img_with_boxes = draw_boxes(img_with_boxes, preds_img)
    cv2.imshow('detections', img_with_boxes)
    # Sometimes the first image shows as black. This seems to be some issue when displaying only, the
    # image itself is fine. If I call waitKey with 0 or with a very large number, the image appears fine.
    cv2.waitKey(1)