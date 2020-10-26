import unittest
from config.train_config import TrainConfiguration
import ssd
from BoundingBoxes import BoundingBox, PredictedBox, boxes_are_equal
import numpy as np
import tensorflow as tf

class SSDTests(unittest.TestCase):

    def test_decode_zeros(self):
        # Decode ground truth where the encoded coordinates are zero. This means that they must
        # coincide with the default boxes.
        args = TrainConfiguration()
        nclasses = 5
        ssd.initialize(nclasses, args.ssd_config)
        tolerance = 1e-6
        labels_enc = np.zeros((ssd.g_nboxes, 6))
        labels_enc[:, 4] = 1
        labels_dec = ssd.decode_gt(labels_enc, remove_duplicated=False)
        for map in ssd.g_feature_maps:
            for anc_idx in range(map.nanchors):
                for row in range(map.grid_size):
                    for col in range(map.grid_size):
                        coords = map.anchor_boxes[anc_idx][row, col, :]
                        # xmin = map.grid[row] - map.widths[anc_idx] / 2.0
                        # ymin = map.grid[col] - map.heights[anc_idx] / 2.0
                        # coords = [xmin, ymin, map.widths[anc_idx], map.heights[anc_idx]]
                        default_box = BoundingBox(coords, 0)
                        found = False
                        for label in labels_dec:
                            # label.print()
                            if boxes_are_equal(default_box, label, tolerance):
                                found = True
                                break
                        if not found:
                            print('Box NOT found:')
                            default_box.print()
                            self.assertTrue(False)


    def test_encode_decode_box(self):
        args = TrainConfiguration()
        nclasses = 5
        ssd.initialize(nclasses, args.ssd_config)
        tolerance = 1e-6
        nattemps = 10
        for _ in range(nattemps):
            xmin = np.random.rand()
            ymin = np.random.rand()
            width = (1 - xmin) * np.random.rand()
            height = (1 - ymin) * np.random.rand()
            box = [xmin, ymin, width, height]
            box_orig_array = np.array(box)
            for feat_map in ssd.g_feature_maps:
                for row in range(feat_map.grid_size):
                    for col in range(feat_map.grid_size):
                        for anc_idx in range(feat_map.nanchors):
                            xc_enc, yc_enc, w_enc, h_enc = ssd.encode_box(box, feat_map, row, col, anc_idx)
                            box_enc = [xc_enc, yc_enc, w_enc, h_enc]
                            xmin, ymin, width, height = ssd.decode_box(box_enc, feat_map, row, col, anc_idx)
                            box_dec = [xmin, ymin, width, height]
                            box_dec_array = np.array(box_dec)
                            self.assertTrue(np.sum(np.square(box_orig_array - box_dec_array)) < tolerance)

    def test_encode_decode_gt(self):
        args = TrainConfiguration()
        nclasses = 5
        ssd.initialize(nclasses, args.ssd_config)
        tolerance = 1e-6
        nlabels = 10
        gt_boxes = []
        for _ in range(nlabels):
            xmin = np.random.rand()
            ymin = np.random.rand()
            width = (1 - xmin) * np.random.rand()
            height = (1 - ymin) * np.random.rand()
            coords = [xmin, ymin, width, height]
            classid = np.random.randint(nclasses)
            gt_boxes.append(BoundingBox(coords, classid))
        encoded_labels = ssd.encode_gt(gt_boxes)
        decoded_labels = ssd.decode_gt(encoded_labels)
        self.assertTrue(len(gt_boxes) == len(decoded_labels), 'Different number of decoded boxes and original ones')
        for box_dec in decoded_labels:
            coords_dec = np.array(box_dec.get_coords())
            found = False
            for box_orig in gt_boxes:
                coords_orig = np.array(box_orig.get_coords())
                if box_orig.classid == box_dec.classid:
                    if np.sum(np.square(coords_orig - coords_dec)) < tolerance:
                        found = True
                        break
            if not found:
                print('Original boxes:')
                for box_orig in gt_boxes:
                    print(str(box_orig.get_coords()) + ' - class id: ' + str(box_orig.classid))
                print('Decoded box:')
                print(str(box_dec.get_coords()) + ' - class id: ' + str(box_dec.classid))
            self.assertTrue(found, 'Decoded box does not match any original box')

    def test_loss_exact_prediction(self):
        args = TrainConfiguration()
        args.batch_size = 4
        nclasses = 20
        ssd.initialize(nclasses, args.ssd_config)
        tolerance = 1e-6
        # Graph structure:
        negative_ratio = 3
        net_output = tf.placeholder(dtype=tf.float32, shape=(args.batch_size, ssd.g_nboxes, 4 + ssd.g_nclasses + 1))
        labels = tf.placeholder(dtype=tf.float32, shape=(args.batch_size, ssd.g_nboxes, 6))
        loss = ssdloss.ssdloss(net_output, labels, negative_ratio, args.batch_size)
        # Create boxes:
        labels_np = np.zeros((args.batch_size, ssd.g_nboxes, 6))
        for b in range(args.batch_size):
            nlabels = np.random.randint(20)
            gt_boxes = []
            for _ in range(nlabels):
                xmin = np.random.rand()
                ymin = np.random.rand()
                width = (1 - xmin) * np.random.rand()
                height = (1 - ymin) * np.random.rand()
                coords = [xmin, ymin, width, height]
                classid = np.random.randint(nclasses)
                gt_boxes.append(BoundingBox(coords, classid))
            encoded_labels = ssd.encode_gt(gt_boxes)
            labels_np[b, :, :] = encoded_labels
        # Make net output the same as the labels, to ensure loss zero:
        net_output_np = np.zeros((args.batch_size, ssd.g_nboxes, 4 + ssd.g_nclasses + 1), dtype=np.float32)
        for b in range(args.batch_size):
            for i in range(ssd.g_nboxes):
                if labels_np[b, i, 4] > 0.5:
                    net_output_np[b, i, :4] = labels_np[b, i, :4]
                    classid = int(labels_np[b, i, 5])
                    net_output_np[b, i, 4:] = -1e6
                    net_output_np[b, i, 4 + classid] = 1e6
                else:
                    net_output_np[b, i, 4:] = -1e6
                    net_output_np[b, i, 4 + ssd.g_nclasses] = 1e0 + np.random.rand() * 1e-1
        with tf.Session() as sess:
            computed_loss = loss.eval(feed_dict={labels: labels_np, net_output: net_output_np})
            print('computed_loss: ' + str(computed_loss))
            self.assertTrue(computed_loss < tolerance)

    def test_loss_missing_unmatched(self):
        args = TrainConfiguration()
        args.batch_size = 4
        nclasses = 20
        ssd.initialize(nclasses, args.ssd_config)
        tolerance = 1e-6
        # Graph structure:
        negative_ratio = 3
        net_output = tf.placeholder(dtype=tf.float32, shape=(args.batch_size, ssd.g_nboxes, 4 + ssd.g_nclasses + 1))
        labels = tf.placeholder(dtype=tf.float32, shape=(args.batch_size, ssd.g_nboxes, 6))
        loss = ssdloss.ssdloss(net_output, labels, negative_ratio, args.batch_size)
        # Create boxes:
        labels_np = np.zeros((args.batch_size, ssd.g_nboxes, 6))
        for b in range(args.batch_size):
            nlabels = np.random.randint(20)
            gt_boxes = []
            for _ in range(nlabels):
                xmin = np.random.rand()
                ymin = np.random.rand()
                width = (1 - xmin) * np.random.rand()
                height = (1 - ymin) * np.random.rand()
                coords = [xmin, ymin, width, height]
                classid = np.random.randint(nclasses)
                gt_boxes.append(BoundingBox(coords, classid))
            encoded_labels = ssd.encode_gt(gt_boxes)
            labels_np[b, :, :] = encoded_labels
        # Make net output the same as the labels, to ensure loss zero:
        net_output_np = np.zeros((args.batch_size, ssd.g_nboxes, 4 + ssd.g_nclasses + 1), dtype=np.float32)
        for b in range(args.batch_size):
            for i in range(ssd.g_nboxes):
                if labels_np[b, i, 4] > 0.5:
                    net_output_np[b, i, :4] = labels_np[b, i, :4]
                    classid = int(labels_np[b, i, 5])
                    net_output_np[b, i, 4:] = -1e6
                    net_output_np[b, i, 4 + classid] = 1e6
                else:
                    for cl in range(ssd.g_nclasses):
                        net_output_np[b, i, 4 + cl] = -1e0 + np.random.rand() * 1e-1
                    if np.random.rand() < 0.9:
                        net_output_np[b, i, 4 + ssd.g_nclasses] = 1e0 + np.random.rand() * 1e-1
                    else:
                        net_output_np[b, i, 4 + np.random.randint(ssd.g_nclasses)] = 1e0 + np.random.rand() * 1e-1
        with tf.Session() as sess:
            computed_loss = loss.eval(feed_dict={labels: labels_np, net_output: net_output_np})
            print('computed_loss: ' + str(computed_loss))
            self.assertTrue(computed_loss > tolerance)

    def test_loss_no_gt(self):
        pass



if __name__ == '__main__':
    unittest.TestLoader().loadTestsFromTestCase(SSDTests)


