import numpy as np
import random
import tensorflow as tf
import os
import shutil
from sys import stdout

random.seed(0)
np.random.seed(0)

from parallel_reading import AsyncParallelReader, ReaderOpts
from loss import SSDLoss
from model import build_model, build_anchors, load_vgg16_weigths
import mean_ap
import tools
import drawing

def lr_schedule(step):
    if step < 60000:
        return 1e-3
    if step < 80000:
        return 1e-4
    else:
        return 1e-5

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

nepochs = 200

period_epochs_check_val = 5
period_batches_display = 20


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


@tf.function
def val_step_fum(batch_imgs, batch_gt):
    net_output = model(batch_imgs, training=True)
    loss_value = loss(batch_gt, net_output)
    loss_value += sum(model.losses)
    return loss_value, net_output

# Refers to the epoch on which the validation loss was best. Used to keep the best model:
best_epoch_idx = -1
best_val_loss = np.inf

checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

checkpoint_to_load = None
# checkpoint_to_load = 'ckpts/ckpt_4'
if checkpoint_to_load is not None:
    read_result = checkpoint.read(checkpoint_to_load)
    read_result.assert_existing_objects_matched()

reader_ops = ReaderOpts(voc_path, nclasses, anchors, img_size, batch_size, nworkers)
with AsyncParallelReader(reader_ops, 'train') as train_reader, \
     AsyncParallelReader(reader_ops, 'val') as val_reader:
    train_step = -1
    for epoch in range(nepochs):
        print("\nStart epoch ", epoch + 1)
        optimizer.learning_rate = lr_schedule(max(train_step, 0))
        # Training:
        for batch_idx in range(train_reader.nbatches):
            train_step += 1
            batch_imgs, batch_gt, batch_gt_raw, names = train_reader.get_batch()
            loss_value, net_output = train_step_fun(
                batch_imgs, batch_gt, tf.cast(tf.convert_to_tensor(train_step), tf.int64))
            stdout.write("\rbatch %d/%d, loss: %.2e    " % (batch_idx + 1, train_reader.nbatches, loss_value.numpy()))
            stdout.flush()
            train_summary_writer.flush()

            if (batch_idx + 1) % period_batches_display == 0:
                drawing.display_gt_and_preds(net_output, batch_imgs, batch_gt_raw, batch_gt, anchors, nclasses)
        stdout.write('\n')

        # Evaluation:
        if (epoch + 1) % period_epochs_check_val == 0:
            print('Running evaluation')
            val_loss = 0.0
            all_batches_gt_raw = []
            all_batches_net_output = []
            for batch_idx in range(val_reader.nbatches):
                batch_imgs, batch_gt, batch_gt_raw, names = val_reader.get_batch()
                loss_value, net_output = val_step_fum(batch_imgs, batch_gt)
                val_loss += loss_value

                all_batches_gt_raw.append(batch_gt_raw)
                all_batches_net_output.append(net_output.numpy())

                stdout.write("\rbatch %d/%d, loss: %.2e    " %
                             (batch_idx + 1, val_reader.nbatches, loss_value.numpy()))
                stdout.flush()

                if (batch_idx + 1) % period_batches_display == 0:
                    drawing.display_gt_and_preds(net_output, batch_imgs, batch_gt_raw, batch_gt, anchors, nclasses)
            stdout.write('\n')
            val_loss /= float(val_reader.nbatches)
            mAP = mean_ap.compute_map(all_batches_net_output, all_batches_gt_raw, anchors)
            print('mAP: %.4f' % mAP)
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


