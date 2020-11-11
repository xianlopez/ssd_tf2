import numpy as np
import random
import tensorflow as tf
import os
import shutil
import cv2
from sys import stdout

random.seed(0)
np.random.seed(0)

from parallel_reading import AsyncParallelReader, image_means, ReaderOpts
from loss import SSDLoss
from model import build_model, build_anchors, load_vgg16_weigths
import mean_ap
import tools
from decoding import decode_preds, decode_gt_batch

def lr_schedule(step):
    if step < 60000:
        return 1e-3
    else:
        return 1e-4

nclasses = 20
img_size = 300

model = build_model()
load_vgg16_weigths(model)
model.build((None, img_size, img_size, 3))
model.summary()

anchors = build_anchors(model)

loss = SSDLoss()

optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule(0), momentum=0.9)

model.compile(loss=loss, optimizer=optimizer)

voc_path = '/home/xian/datasets/VOCdevkit'
batch_size = 32
nworkers = 8

# current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
# train_log_dir = 'logs/' + current_time + '/train'
train_log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs')
if os.path.exists(train_log_dir):
    shutil.rmtree(train_log_dir)
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
train_summary_writer.set_as_default()

nepochs = 1000

period_epochs_check_val = 5
period_batches_display = 20


@tf.function
def train_step(batch_imgs, batch_gt, step):
    with tf.GradientTape() as tape:
        net_output = model(batch_imgs, training=True)
        loss_value = loss(batch_gt, net_output)
        loss_value += sum(model.losses)
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', loss_value, step=step)
    return loss_value, net_output


@tf.function
def val_step(batch_imgs, batch_gt):
    net_output = model(batch_imgs, training=True)
    loss_value = loss(batch_gt, net_output)
    loss_value += sum(model.losses)
    return loss_value, net_output

# Refers to the epoch on which the validation loss was best. Used to keep the best model:
best_epoch_idx = -1
best_val_loss = np.inf

checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

reader_ops = ReaderOpts(voc_path, nclasses, anchors, img_size, batch_size, nworkers)
with AsyncParallelReader(reader_ops, 'train') as train_reader, \
     AsyncParallelReader(reader_ops, 'val') as val_reader:
    step = -1
    for epoch in range(nepochs):
        print("\nStart epoch ", epoch + 1)
        optimizer.learning_rate = lr_schedule(max(step, 0))
        # Training:
        for batch_idx in range(train_reader.nbatches):
            step += 1
            batch_imgs, batch_gt, batch_gt_raw, names = train_reader.get_batch()
            loss_value, net_output = train_step(batch_imgs, batch_gt, tf.cast(tf.convert_to_tensor(step), tf.int64))
            stdout.write("\rbatch %d/%d, loss: %.2e    " % (batch_idx + 1, train_reader.nbatches, loss_value.numpy()))
            stdout.flush()
            train_summary_writer.flush()

            if (batch_idx + 1) % period_batches_display == 0:
                predictions = decode_preds(net_output.numpy(), anchors, nclasses)
                mAP = mean_ap.mean_ap_on_batch(predictions, batch_gt_raw, nclasses)
                stdout.write('\n')
                print('mAP = %.4f' % mAP)
                print('Showing image: ' + names[0])
                img = batch_imgs[0, ...]
                img = img + image_means
                img_gt = img.copy()
                gt = np.take(batch_gt_raw[0], np.where(batch_gt_raw[0][:, 4] != nclasses)[0], axis=0)  # (ngt, 5)
                img_gt = tools.draw_boxes(img_gt, gt)
                cv2.imshow('image with GT', img_gt)
                img_gt_dec = img.copy()
                gt_dec = decode_gt_batch(batch_gt, anchors)[0]  # (ngt_dec, 5)
                img_gt_dec = tools.draw_boxes(img_gt_dec, gt_dec)
                cv2.imshow('image with decoded GT', img_gt_dec)
                preds_img = predictions[0]
                print(str(len(preds_img)) + ' predictions')
                img_with_boxes = img.copy()
                img_with_boxes = tools.draw_boxes(img_with_boxes, preds_img)
                cv2.imshow('detections', img_with_boxes)
                # Sometimes the first image shows as black. This seems to be some issue when displaying only, the
                # image itself is fine. If I call waitKey with 0 or with a very large number, the image appears fine.
                cv2.waitKey(1)
        stdout.write('\n')

        # Evaluation:
        if (epoch + 1) % period_epochs_check_val == 0:
            print('Running evaluation')
            mAP_sum = 0.0
            val_loss = 0.0
            for batch_idx in range(val_reader.nbatches):
                step += 1
                batch_imgs, batch_gt, batch_gt_raw, names = val_reader.get_batch()
                loss_value, net_output = val_step(batch_imgs, batch_gt)
                val_loss += loss_value

                predictions = decode_preds(net_output.numpy(), anchors, nclasses)
                mAP_batch = mean_ap.mean_ap_on_batch(predictions, batch_gt_raw, nclasses)
                stdout.write("\rbatch %d/%d, loss: %.2e, mAP: %.4f    " %
                             (batch_idx + 1, val_reader.nbatches, loss_value.numpy(), mAP_batch))
                stdout.flush()
                mAP_sum += mAP_batch

                if (batch_idx + 1) % period_batches_display == 0:
                    stdout.write('\n')
                    print('Showing image: ' + names[0])
                    img = batch_imgs[0, ...]
                    img = img + image_means
                    img_gt = img.copy()
                    gt = np.take(batch_gt_raw[0], np.where(batch_gt_raw[0][:, 4] != nclasses)[0], axis=0)  # (ngt, 5)
                    img_gt = tools.draw_boxes(img_gt, gt)
                    cv2.imshow('image with GT', img_gt)
                    img_gt_dec = img.copy()
                    gt_dec = decode_gt_batch(batch_gt, anchors)[0]  # (ngt_dec, 5)
                    img_gt_dec = tools.draw_boxes(img_gt_dec, gt_dec)
                    cv2.imshow('image with decoded GT', img_gt_dec)
                    preds_img = predictions[0]
                    print(str(len(preds_img)) + ' predictions')
                    img_with_boxes = img.copy()
                    img_with_boxes = tools.draw_boxes(img_with_boxes, preds_img)
                    cv2.imshow('detections', img_with_boxes)
                    # Sometimes the first image shows as black. This seems to be some issue when displaying only, the
                    # image itself is fine. If I call waitKey with 0 or with a very large number, the image appears fine.
                    cv2.waitKey(1)

            stdout.write('\n')
            val_loss /= float(val_reader.nbatches)
            mAP = mAP_sum / float(val_reader.nbatches)
            print('Mean mAP on the whole epoch: %.4f' % mAP)

        else:
            val_loss = np.inf

        # Save model:
        print('Saving model')
        checkpoint.write('ckpts/ckpt_' + str(epoch))
        if val_loss < best_val_loss:
            # Erase last epoch's and previous "best" checkpoint:
            if best_epoch_idx >= 0:
                tools.delete_checkpoint('ckpts/ckpt_' + str(best_epoch_idx))
            if epoch > 0:
                tools.delete_checkpoint('ckpts/ckpt_' + str(epoch - 1))
            best_epoch_idx = epoch
            best_val_loss = val_loss
        elif best_val_loss != epoch - 1 and epoch > 0:
            # Erase last epoch's checkpoint:
            tools.delete_checkpoint('ckpts/ckpt_' + str(epoch - 1))


