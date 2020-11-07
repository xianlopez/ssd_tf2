import tensorflow as tf

model1 = tf.keras.applications.VGG16(include_top=True)

print('model1')
for i in range(len(model1.layers)):
    l = model1.layers[i]
    text = str(i) + ' ' + l._name
    for w in l.weights:
        text += " " + str(w.shape)
    print(text)

fc1 = model1.layers[20]
assert fc1._name == 'fc1'
kernel1 = tf.reshape(fc1.weights[0], (7, 7, 512, 4096))
kernel1_sub = kernel1[0:7:3, 0:7:3, :, 0:4096:4]
bias1_sub = fc1.weights[1][0:4096:4]
def kernel1_init(shape, dtype=None):
    return kernel1_sub
def bias1_init(shape, dtype=None):
    return bias1_sub

fc2 = model1.layers[21]
assert fc2._name == 'fc2'
kernel2_sub = fc2.weights[0][0:4096:4, 0:4096:4]  # (1024, 1024)
kernel2_sub = tf.expand_dims(kernel2_sub, axis=0)  # (1, 1024, 1024)
kernel2_sub = tf.expand_dims(kernel2_sub, axis=0)  # (1, 1, 1024, 1024)
bias2_sub = fc2.weights[1][0:4096:4]
def kernel2_init(shape, dtype=None):
    return kernel2_sub
def bias2_init(shape, dtype=None):
    return bias2_sub

vgg_no_head = tf.keras.applications.VGG16(include_top=False)
pool5 = vgg_no_head.output
x = tf.keras.layers.Conv2D(1024, 3, name='conv6', kernel_initializer=kernel1_init, bias_initializer=bias1_init)(pool5)
x = tf.keras.layers.Conv2D(1024, 1, name='conv7', kernel_initializer=kernel2_init, bias_initializer=bias2_init)(x)
model2 = tf.keras.Model(inputs=vgg_no_head.inputs, outputs=x)

print('model2')
for i in range(len(model2.layers)):
    l = model2.layers[i]
    text = str(i) + ' ' + l._name
    for w in l.weights:
        text += " " + str(w.shape)
    print(text)

checkpoint = tf.train.Checkpoint(model=model2)
checkpoint.write('weights/vgg16')

print('')
print('some weights:')
print(model2.layers[1].weights[0].numpy()[0, 0, 0, 0])
print(model2.layers[1].weights[1].numpy()[25])
print(model2.layers[2].weights[0].numpy()[2, 2, 63, 63])
print(model2.layers[4].weights[1].numpy()[1])
print(model2.layers[15].weights[0].numpy()[1, 0, 63, 269])
print(model2.layers[16].weights[1].numpy()[510])
print(model2.layers[17].weights[0].numpy()[0, 2, 0, 400])
print(model2.layers[19].weights[0].numpy()[1, 2, 1, 39])
print(model2.layers[19].weights[1].numpy()[789])
print(model2.layers[20].weights[0].numpy()[0, 0, 1023, 1000])

