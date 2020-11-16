import numpy as np
import random
import tensorflow as tf
import os
import shutil
from sys import stdout
from datetime import datetime
import argparse
import ast

from parallel_reading import AsyncParallelReader, ReaderOpts
from loss import SSDLoss
from model import build_model, build_anchors, load_vgg16_weigths
import tools
import drawing
from evaluate import evaluation_loop, build_val_step_fun

nclasses = 20
img_size = 300
voc_path = '/home/xian/datasets/VOCdevkit'


def lr_schedule(current_step, step_lr_pairs, initial_lr):
    prev_starting_step = np.inf
    for starting_step, lr in step_lr_pairs[::-1]:
        assert starting_step < prev_starting_step
        if current_step >= starting_step:
            return lr
    return initial_lr


def build_train_step_fun(model, loss, optimizer, train_summary_writer):
    @tf.function
    def train_step_fun(batch_imgs, batch_gt, step):
        with tf.GradientTape() as tape:
            net_output = model(batch_imgs, training=True)
            loss_value = loss(batch_gt, net_output)
            loss_value += sum(model.losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss_value, step=step)
        return loss_value, net_output
    return train_step_fun


def train(ckpt_idx, batch_size, nworkers, nepochs, period_display, period_evaluate,
          initial_lr, step_lr_pairs):
    model = build_model()
    load_vgg16_weigths(model)
    model.build((None, img_size, img_size, 3))
    model.summary()
    anchors = build_anchors(model)
    loss = SSDLoss()
    optimizer = tf.keras.optimizers.SGD(learning_rate=initial_lr, momentum=0.9)
    model.compile(loss=loss, optimizer=optimizer)

    train_log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs')
    if os.path.exists(train_log_dir):
        shutil.rmtree(train_log_dir)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    train_summary_writer.set_as_default()

    val_step_fun = build_val_step_fun(model, loss)
    train_step_fun = build_train_step_fun(model, loss, optimizer, train_summary_writer)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

    if ckpt_idx:
        checkpoint_to_load = 'ckpts/ckpt_' + str(ckpt_idx)
        read_result = checkpoint.read(checkpoint_to_load)
        read_result.assert_existing_objects_matched()

    # Refers to the epoch on which the validation loss was best. Used to keep the best model:
    best_epoch_idx = -1
    best_val_loss = np.inf

    reader_opts = ReaderOpts(voc_path, nclasses, anchors, img_size, batch_size, nworkers)
    with AsyncParallelReader(reader_opts, 'train') as train_reader, \
         AsyncParallelReader(reader_opts, 'val') as val_reader:
        train_step = -1
        for epoch in range(nepochs):
            print("\nStart epoch ", epoch + 1)
            optimizer.learning_rate = lr_schedule(train_step, step_lr_pairs, initial_lr)
            print('Learning rate: %.2e' % optimizer.learning_rate)
            # Training:
            epoch_start = datetime.now()
            for batch_idx in range(train_reader.nbatches):
                train_step += 1
                batch_imgs, batch_gt, batch_gt_raw, names = train_reader.get_batch()
                loss_value, net_output = train_step_fun(
                    batch_imgs, batch_gt, tf.cast(tf.convert_to_tensor(train_step), tf.int64))
                stdout.write("\rbatch %d/%d, loss: %.2e    " % (batch_idx + 1, train_reader.nbatches, loss_value.numpy()))
                stdout.flush()
                train_summary_writer.flush()

                if (batch_idx + 1) % period_display == 0:
                    drawing.display_gt_and_preds(net_output, batch_imgs, batch_gt_raw, batch_gt, anchors, nclasses)
            stdout.write('\n')
            print('Epoch computed in ' + str(datetime.now() - epoch_start))

            # Evaluation:
            if (epoch + 1) % period_evaluate == 0:
                val_loss, mAP = evaluation_loop(val_step_fun, val_reader, period_display, anchors)
            else:
                val_loss = np.inf

            # Save model:
            print('Saving model')
            checkpoint.write('ckpts/ckpt_' + str(epoch))
            if val_loss < best_val_loss:
                # Erase last epoch's and previous "best" checkpoint:
                if best_epoch_idx >= 0:
                    tools.delete_checkpoint_with_index(best_epoch_idx)
                if epoch > 0:
                    tools.delete_checkpoint_with_index(epoch - 1)
                best_epoch_idx = epoch
                best_val_loss = val_loss
            elif best_val_loss != epoch - 1 and epoch > 0:
                # Erase last epoch's checkpoint:
                tools.delete_checkpoint_with_index(epoch - 1)


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser(description='Evaluate a network')
    parser.add_argument('--ckpt_idx', type=int, help='index of the checkpoint to load the initial weights')
    parser.add_argument('--nworkers', type=int, default=8, help='number of processes to read data')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--nepochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--lr_changes', default='[(60000, 1e-4), (80000, 1e-5)]',
                        help='changes in learning rate, as a list of tuples where the first element is the step from '
                             'which the second one (learning rate) applies')
    parser.add_argument('--period_display', type=int, default=10,
                        help='number of batches between two consecutive displays')
    parser.add_argument('--period_evaluate', type=int, default=5,
                        help='number of epochs between two consecutive evaluations')
    args = parser.parse_args()

    step_lr_pairs = ast.literal_eval(args.lr_changes)

    train(args.ckpt_idx, args.batch_size, args.nworkers, args.nepochs, args.period_display,
          args.period_evaluate, args.learning_rate, step_lr_pairs)
