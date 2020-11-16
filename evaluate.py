import numpy as np
import random
import tensorflow as tf
from sys import stdout
from datetime import datetime
import argparse

from parallel_reading import AsyncParallelReader, ReaderOpts
from loss import SSDLoss
from model import build_model, build_anchors, load_vgg16_weigths
import mean_ap
import drawing


nclasses = 20
img_size = 300
voc_path = '/home/xian/datasets/VOCdevkit'


def build_val_step_fun(model, loss):
    @tf.function
    def val_step_fun(batch_imgs, batch_gt):
        net_output = model(batch_imgs, training=True)
        loss_value = loss(batch_gt, net_output)
        loss_value += sum(model.losses)
        return loss_value, net_output
    return val_step_fun


def evaluation_loop(val_step_fun, val_reader, period_display, anchors):
    print('Running evaluation')
    evaluation_start = datetime.now()
    val_loss = 0.0
    all_batches_gt_raw = []
    all_batches_net_output = []
    for batch_idx in range(val_reader.nbatches):
        batch_imgs, batch_gt, batch_gt_raw, names = val_reader.get_batch()
        loss_value, net_output = val_step_fun(batch_imgs, batch_gt)
        val_loss += loss_value

        all_batches_gt_raw.append(batch_gt_raw)
        all_batches_net_output.append(net_output.numpy())

        stdout.write("\rbatch %d/%d, loss: %.2e    " %
                     (batch_idx + 1, val_reader.nbatches, loss_value.numpy()))
        stdout.flush()

        if (batch_idx + 1) % period_display == 0:
            drawing.display_gt_and_preds(net_output, batch_imgs, batch_gt_raw, batch_gt, anchors, nclasses)
    stdout.write('\n')
    val_loss /= float(val_reader.nbatches)
    mAP = mean_ap.compute_map(all_batches_net_output, all_batches_gt_raw, anchors)
    print('mAP: %.4f' % mAP)
    print('loss: %.4e' % val_loss)
    print('Evaluation computed in ' + str(datetime.now() - evaluation_start))
    return val_loss, mAP


def evaluate(ckpt_idx, period_display, nworkers, batch_size):
    model = build_model()
    load_vgg16_weigths(model)
    model.build((None, img_size, img_size, 3))
    model.summary()
    anchors = build_anchors(model)
    loss = SSDLoss()
    model.compile(loss=loss)

    checkpoint = tf.train.Checkpoint(model=model)

    checkpoint_to_load = 'ckpts/ckpt_' + str(ckpt_idx)
    read_result = checkpoint.read(checkpoint_to_load)
    read_result.assert_existing_objects_matched()

    val_step_fun = build_val_step_fun(model, loss)

    reader_opts = ReaderOpts(voc_path, nclasses, anchors, img_size, batch_size, nworkers)
    with AsyncParallelReader(reader_opts, 'val') as val_reader:
        evaluation_loop(val_step_fun, val_reader, period_display, anchors)


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser(description='Evaluate a network')
    parser.add_argument('--ckpt_idx', type=int, required=True, help='index of the checkpoint to load')
    parser.add_argument('--nworkers', type=int, default=8, help='number of processes to read data')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--period_display', type=int, default=10,
                        help='number of batches between two consecutive displays')
    args = parser.parse_args()

    evaluate(args.ckpt_idx, args.period_display, args.nworkers, args.batch_size)

