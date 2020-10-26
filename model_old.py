import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model, Input, utils

aspect_ratios = [1.0, 0.5, 2.0]
nclasses = 20


class PredictionHead(layers.Layer):
    def __init__(self):
        super(PredictionHead, self).__init__(name="PredictionHead")
        self.num_ouptuts = len(aspect_ratios) * (4 + nclasses + 1)  # 4 for the coordinates, 1 for background.
        self.conv = layers.Conv2D(self.num_ouptuts, 3, kernel_initializer=tf.initializers.he_normal(), padding='same')
        self.reshape = layers.Reshape((-1, 4 + nclasses + 1))
        # self.nanchors_in_layer = grid_size * grid_size * len(aspect_ratios)

    def call(self, x):
        x = self.conv(x)
        x = self.reshape(x)
        print("x shape after reshape:")
        print(x.shape)
        # x = tf.reshape(x, [-1, self.nanchors_in_layer, 4 + nclasses + 1])
        return x  # (?, nanchors_in_layer, 4 + nclasses + 1)


def vgg16_body(img_size):
    vgg = Sequential([Input(shape=[img_size, img_size, 3])])
    vgg.add(layers.Conv2D(64, 3, activation='relu', padding='same', name='conv1_1'))
    vgg.add(layers.Conv2D(64, 3, activation='relu', padding='same', name='conv1_2'))
    vgg.add(layers.MaxPool2D(name='pool1'))  # (?, 150, 150, 64)
    vgg.add(layers.Conv2D(128, 3, activation='relu', padding='same', name='conv2_1'))
    vgg.add(layers.Conv2D(128, 3, activation='relu', padding='same', name='conv2_2'))
    vgg.add(layers.MaxPool2D(name='pool2')) # (?, 75, 75, 128)
    vgg.add(layers.Conv2D(256, 3, activation='relu', padding='same', name='conv3_1'))
    vgg.add(layers.Conv2D(256, 3, activation='relu', padding='same', name='conv3_2'))
    vgg.add(layers.Conv2D(256, 3, activation='relu', padding='same', name='conv3_3'))
    vgg.add(layers.MaxPool2D(padding='same', name='pool3'))  # (?, 38, 38, 256)
    vgg.add(layers.Conv2D(512, 3, activation='relu', padding='same', name='conv4_1'))
    vgg.add(layers.Conv2D(512, 3, activation='relu', padding='same', name='conv4_2'))
    vgg.add(layers.Conv2D(512, 3, activation='relu', padding='same', name='conv4_3'))
    vgg.add(layers.MaxPool2D(name='pool4'))  # (?, 19, 19, 512)
    vgg.add(layers.Conv2D(512, 3, activation='relu', padding='same', name='conv5_1'))
    vgg.add(layers.Conv2D(512, 3, activation='relu', padding='same', name='conv5_2'))
    vgg.add(layers.Conv2D(512, 3, activation='relu', padding='same', name='conv5_3'))
    vgg.add(layers.MaxPool2D(pool_size=(3, 3), padding='same', strides=(1, 1), name='pool5'))  # (?, 19, 19, 512)

    conv4_3 = vgg.get_layer("conv4_3").output  # (?, 38, 38, 256)
    pool5 = vgg.output  # (?, 19, 19, 512)

    return Model(inputs=[vgg.inputs], outputs=[conv4_3, pool5], name='vgg16')


def extra_blocks(vgg_pool5):
    kernel_init = tf.initializers.he_normal()
    model = Sequential(vgg_pool5)

    model.add(layers.Conv2D(1024, 3, dilation_rate=6, activation='relu', kernel_initializer=kernel_init,
                      padding='same', name='conv6'))  # (?, 19, 19, 1024)
    model.add(layers.Conv2D(1024, 1, activation='relu', kernel_initializer=kernel_init,
                      padding='same', name='conv7'))  # (?, 19, 19, 1024)

    model.add(layers.Conv2D(256, 1, activation='relu', kernel_initializer=kernel_init,
                      padding='same', name='conv8_1'))  # (?, 19, 19, 256)
    model.add(layers.Conv2D(512, 3, activation='relu', strides=2, kernel_initializer=kernel_init,
                      padding='same', name='conv8_2'))  # (?, 10, 10, 512)

    model.add(layers.Conv2D(128, 1, activation='relu', kernel_initializer=kernel_init,
                      padding='same', name='conv9_1'))  # (?, 10, 10, 128)
    model.add(layers.Conv2D(256, 3, activation='relu', strides=2, kernel_initializer=kernel_init,
                      padding='same', name='conv9_2'))  # (?, 5, 5, 256)

    model.add(layers.Conv2D(128, 1, activation='relu', kernel_initializer=kernel_init,
                      padding='same', name='conv10_1'))  # (?, 5, 5, 128)
    model.add(layers.Conv2D(256, 3, activation='relu', kernel_initializer=kernel_init,
                      padding='valid', name='conv10_2'))  # (?, 3, 3, 256)

    model.add(layers.Conv2D(128, 1, activation='relu', kernel_initializer=kernel_init,
                      padding='same', name='conv11_1'))  # (?, 3, 3, 128)
    model.add(layers.Conv2D(256, 3, activation='relu', kernel_initializer=kernel_init,
                      padding='valid', name='conv11_2'))  # (?, 1, 1, 256)

    conv7 = model.get_layer("conv7").output  # (?, 19, 19, 1024)
    conv8_2 = model.get_layer("conv8_2").output  # (?, 10, 10, 512)
    conv9_2 = model.get_layer("conv9_2").output  # (?, 5, 5, 256)
    conv10_2 = model.get_layer("conv10_2").output  # (?, 3, 3, 256)
    conv11_2 = model.get_layer("conv11_2").output  # (?, 1, 1, 256)

    return Model(inputs=[model.inputs], outputs=[conv7, conv8_2, conv9_2, conv10_2, conv11_2], name='new_blocks')


class SSD(Model):
    def __init__(self, img_size):
        super(SSD, self).__init__(name="SSD")

        self.vgg = vgg16_body(img_size)
        self.extra_blocks = extra_blocks(self.vgg.output[1])
        self.head1 = PredictionHead()
        self.head2 = PredictionHead()
        self.head3 = PredictionHead()
        self.head4 = PredictionHead()
        self.head5 = PredictionHead()
        self.head6 = PredictionHead()

    def call(self, image):
        conv4_3, pool5 = self.vgg(image)
        conv7, conv8_2, conv9_2, conv10_2, conv11_2 = self.extra_blocks(pool5)
        pred1 = self.head1(conv4_3)  # (?, x1, 4 + nclasses + 1)
        pred2 = self.head2(conv7)  # (?, x2, 4 + nclasses + 1)
        pred3 = self.head3(conv8_2)  # (?, x3, 4 + nclasses + 1)
        pred4 = self.head4(conv9_2)  # (?, x4, 4 + nclasses + 1)
        pred5 = self.head5(conv10_2)  # (?, x5, 4 + nclasses + 1)
        pred6 = self.head6(conv11_2)  # (?, x6, 4 + nclasses + 1)
        return tf.concat([pred1, pred2, pred3, pred4, pred5, pred6], axis=1)  # (?, nanchors, 4 + nclasses + 1)

    def build_anchors(self):
        pass


# if True:
if __name__ == "__main__":
    model = SSD(300)
    model.build((None, 300, 300, 3))
    model.summary()

    import numpy as np
    x = np.zeros((16, 300, 300, 3), dtype=np.float32)
    output = model(x)
    print('output.shape')
    print(output.shape)


