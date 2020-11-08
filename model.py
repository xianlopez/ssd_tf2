import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model, Input, utils
import numpy as np
from tensorflow.keras.regularizers import l2

l2_reg = 5e-4
aspect_ratios = [1.0, 0.5, 2.0]
nclasses = 20
img_size = 300

anchors_sizes = {
    'head1': 0.15,
    'head2': 0.25,
    'head3': 0.35,
    'head4': 0.50,
    'head5': 0.7,
    'head6': 0.85
}


class ChannelsL2Normalization(layers.Layer):
    def __init__(self, **kwargs):
        super(ChannelsL2Normalization, self).__init__(**kwargs)
        self.w = tf.Variable(initial_value=20, dtype=tf.float32)

    def call(self, x):
        x = tf.math.l2_normalize(x, axis=-1)
        x = x * self.w
        return x


class PredictionHead(layers.Layer):
    def __init__(self, normalize=False, **kwargs):
        super(PredictionHead, self).__init__(**kwargs)
        if normalize:
            name = self.name + '_l2norm'
            self.norm_layer = ChannelsL2Normalization(name=name)
        else:
            self.norm_layer = None
        self.num_ouptuts = len(aspect_ratios) * (4 + nclasses + 1)  # 4 for the coordinates, 1 for background.
        self.conv = layers.Conv2D(self.num_ouptuts, 3, kernel_initializer=tf.initializers.he_normal(),
                                  kernel_regularizer=l2(l2=l2_reg), bias_regularizer=l2(l2=l2_reg), padding='same')
        self.reshape = layers.Reshape((-1, 4 + nclasses + 1))

    def call(self, x):
        if self.norm_layer:
            x = self.norm_layer(x)
        x = self.conv(x)
        x = self.reshape(x)
        return x  # (?, nanchors_in_layer, 4 + nclasses + 1)


def build_model():
    # VGG-16 blocks:
    inputs = Input(shape=(img_size, img_size, 3))
    x = layers.Conv2D(64, 3, activation='relu', padding='same', name='conv1_1',
                      kernel_regularizer=l2(l2=l2_reg), bias_regularizer=l2(l2=l2_reg))(inputs)
    x = layers.Conv2D(64, 3, activation='relu', padding='same', name='conv1_2',
                      kernel_regularizer=l2(l2=l2_reg), bias_regularizer=l2(l2=l2_reg))(x)
    x = layers.MaxPool2D(name='pool1')(x)  # (?, 150, 150, 64)
    x = layers.Conv2D(128, 3, activation='relu', padding='same', name='conv2_1',
                      kernel_regularizer=l2(l2=l2_reg), bias_regularizer=l2(l2=l2_reg))(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same', name='conv2_2',
                      kernel_regularizer=l2(l2=l2_reg), bias_regularizer=l2(l2=l2_reg))(x)
    x = layers.MaxPool2D(name='pool2')(x)  # (?, 75, 75, 128)
    x = layers.Conv2D(256, 3, activation='relu', padding='same', name='conv3_1',
                      kernel_regularizer=l2(l2=l2_reg), bias_regularizer=l2(l2=l2_reg))(x)
    x = layers.Conv2D(256, 3, activation='relu', padding='same', name='conv3_2',
                      kernel_regularizer=l2(l2=l2_reg), bias_regularizer=l2(l2=l2_reg))(x)
    x = layers.Conv2D(256, 3, activation='relu', padding='same', name='conv3_3',
                      kernel_regularizer=l2(l2=l2_reg), bias_regularizer=l2(l2=l2_reg))(x)
    x = layers.MaxPool2D(padding='same', name='pool3')(x)  # (?, 38, 38, 256)
    x = layers.Conv2D(512, 3, activation='relu', padding='same', name='conv4_1',
                      kernel_regularizer=l2(l2=l2_reg), bias_regularizer=l2(l2=l2_reg))(x)
    x = layers.Conv2D(512, 3, activation='relu', padding='same', name='conv4_2',
                      kernel_regularizer=l2(l2=l2_reg), bias_regularizer=l2(l2=l2_reg))(x)
    conv4_3 = layers.Conv2D(512, 3, activation='relu', padding='same', name='conv4_3')(x)
    x = layers.MaxPool2D(name='pool4')(conv4_3)  # (?, 19, 19, 512)
    x = layers.Conv2D(512, 3, activation='relu', padding='same', name='conv5_1',
                      kernel_regularizer=l2(l2=l2_reg), bias_regularizer=l2(l2=l2_reg))(x)
    x = layers.Conv2D(512, 3, activation='relu', padding='same', name='conv5_2',
                      kernel_regularizer=l2(l2=l2_reg), bias_regularizer=l2(l2=l2_reg))(x)
    x = layers.Conv2D(512, 3, activation='relu', padding='same', name='conv5_3',
                      kernel_regularizer=l2(l2=l2_reg), bias_regularizer=l2(l2=l2_reg))(x)
    x = layers.MaxPool2D(pool_size=(3, 3), padding='same', strides=(1, 1), name='pool5')(x)  # (?, 19, 19, 512)

    # Extra blocks:
    kernel_init = tf.initializers.he_normal()
    x = layers.Conv2D(1024, 3, dilation_rate=6, activation='relu', kernel_initializer=kernel_init,
                      kernel_regularizer=l2(l2=l2_reg), bias_regularizer=l2(l2=l2_reg),
                      padding='same', name='conv6')(x)  # (?, 19, 19, 1024)
    conv7 = layers.Conv2D(1024, 1, activation='relu', kernel_initializer=kernel_init,
                      kernel_regularizer=l2(l2=l2_reg), bias_regularizer=l2(l2=l2_reg),
                      padding='same', name='conv7')(x)  # (?, 19, 19, 1024)

    x = layers.Conv2D(256, 1, activation='relu', kernel_initializer=kernel_init,
                      kernel_regularizer=l2(l2=l2_reg), bias_regularizer=l2(l2=l2_reg),
                      padding='same', name='conv8_1')(conv7)  # (?, 19, 19, 256)
    conv8_2 = layers.Conv2D(512, 3, activation='relu', strides=2, kernel_initializer=kernel_init,
                      kernel_regularizer=l2(l2=l2_reg), bias_regularizer=l2(l2=l2_reg),
                      padding='same', name='conv8_2')(x)  # (?, 10, 10, 512)

    x = layers.Conv2D(128, 1, activation='relu', kernel_initializer=kernel_init,
                      kernel_regularizer=l2(l2=l2_reg), bias_regularizer=l2(l2=l2_reg),
                      padding='same', name='conv9_1')(conv8_2)  # (?, 10, 10, 128)
    conv9_2 = layers.Conv2D(256, 3, activation='relu', strides=2, kernel_initializer=kernel_init,
                      kernel_regularizer=l2(l2=l2_reg), bias_regularizer=l2(l2=l2_reg),
                      padding='same', name='conv9_2')(x)  # (?, 5, 5, 256)

    x = layers.Conv2D(128, 1, activation='relu', kernel_initializer=kernel_init,
                      kernel_regularizer=l2(l2=l2_reg), bias_regularizer=l2(l2=l2_reg),
                      padding='same', name='conv10_1')(conv9_2)  # (?, 5, 5, 128)
    conv10_2 = layers.Conv2D(256, 3, activation='relu', kernel_initializer=kernel_init,
                      kernel_regularizer=l2(l2=l2_reg), bias_regularizer=l2(l2=l2_reg),
                      padding='valid', name='conv10_2')(x)  # (?, 3, 3, 256)

    x = layers.Conv2D(128, 1, activation='relu', kernel_initializer=kernel_init,
                      kernel_regularizer=l2(l2=l2_reg), bias_regularizer=l2(l2=l2_reg),
                      padding='same', name='conv11_1')(conv10_2)  # (?, 3, 3, 128)
    conv11_2 = layers.Conv2D(256, 3, activation='relu', kernel_initializer=kernel_init,
                      kernel_regularizer=l2(l2=l2_reg), bias_regularizer=l2(l2=l2_reg),
                      padding='valid', name='conv11_2')(x)  # (?, 1, 1, 256)

    # Prediction heads:
    pred1 = PredictionHead(name='head1', normalize=True)(conv4_3)
    pred2 = PredictionHead(name='head2')(conv7)
    pred3 = PredictionHead(name='head3')(conv8_2)
    pred4 = PredictionHead(name='head4')(conv9_2)
    pred5 = PredictionHead(name='head5')(conv10_2)
    pred6 = PredictionHead(name='head6')(conv11_2)

    output = layers.concatenate([pred1, pred2, pred3, pred4, pred5, pred6], axis=1)  # (?, nanchors, 4 + nclasses + 1)

    return Model(inputs=inputs, outputs=output, name="SSD")


def build_anchors(model):
    anchors_all_heads = []
    for head_num in range(1, 7):
        head_name = 'head' + str(head_num)
        head = model.get_layer(head_name)
        anchor_size = anchors_sizes[head_name]
        grid_size = head.input_shape[1]
        grid = np.linspace(0, 1, grid_size + 1)[:grid_size] + 1.0 / (2.0 * grid_size)

        anchors_this_head = np.zeros(shape=(grid_size, grid_size, len(aspect_ratios), 4), dtype=np.float32)
        for ratio_idx in range(len(aspect_ratios)):
            ratio = aspect_ratios[ratio_idx]
            width = anchor_size * np.sqrt(ratio)
            height = anchor_size / np.sqrt(ratio)
            for row in range(grid_size):
                center_y = grid[row]
                y_min = max(0, center_y - height / 2.0)
                y_max = min(1.0, center_y + height / 2.0)
                row_height = y_max - y_min
                for col in range(grid_size):
                    center_x = grid[col]
                    x_min = max(0, center_x - width / 2.0)
                    x_max = min(1.0, center_x + width / 2.0)
                    col_width = x_max - x_min
                    anchors_this_head[row, col, ratio_idx, 0] = x_min
                    anchors_this_head[row, col, ratio_idx, 1] = y_min
                    anchors_this_head[row, col, ratio_idx, 2] = col_width
                    anchors_this_head[row, col, ratio_idx, 3] = row_height
        anchors_this_head = np.reshape(anchors_this_head, [-1, 4])
        anchors_all_heads.append(anchors_this_head)

    anchors = np.concatenate(anchors_all_heads, axis=0)  # (nanchors, 4)
    return anchors


def load_vgg16_weigths(model):
    print('Loading VGG-16 weights...')
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.read('weights/vgg16')
    # We'll assert the loaded was right by probing some particular weigths:
    assert np.abs(model.layers[1].weights[0].numpy()[0, 0, 0, 0] - 0.42947057) < 1e-6
    assert np.abs(model.layers[1].weights[1].numpy()[25] - 0.7446502) < 1e-6
    assert np.abs(model.layers[2].weights[0].numpy()[2, 2, 63, 63] - 0.016082443) < 1e-6
    assert np.abs(model.layers[4].weights[1].numpy()[1] - 0.06440781) < 1e-6
    assert np.abs(model.layers[15].weights[0].numpy()[1, 0, 63, 269] - -0.011236359) < 1e-6
    assert np.abs(model.layers[16].weights[1].numpy()[510] - 0.23517226) < 1e-6
    assert np.abs(model.layers[17].weights[0].numpy()[0, 2, 0, 400] - -0.008714244) < 1e-6
    assert np.abs(model.layers[19].weights[0].numpy()[1, 2, 1, 39] - -0.0012724434) < 1e-6
    assert np.abs(model.layers[19].weights[1].numpy()[789] - 0.15145591) < 1e-6
    assert np.abs(model.layers[20].weights[0].numpy()[0, 0, 1023, 1000] - -0.0027780728) < 1e-6

