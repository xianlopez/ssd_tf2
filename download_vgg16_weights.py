import tensorflow as tf

model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None,
                            classes=1000, classifier_activation='softmax')

for i in range(len(model.layers)):
    l = model.layers[i]
    text = str(i) + ' ' + l._name
    for w in l.weights:
        text += " " + str(w.shape)
    print(text)

print('')
print('some weights:')
print(model.layers[1].weights[0].numpy()[0, 0, 0, 0])
print(model.layers[1].weights[1].numpy()[25])
print(model.layers[2].weights[0].numpy()[2, 2, 63, 63])
print(model.layers[4].weights[1].numpy()[1])
print(model.layers[15].weights[0].numpy()[1, 0, 63, 269])
print(model.layers[16].weights[1].numpy()[510])
print(model.layers[17].weights[0].numpy()[0, 2, 0, 400])

checkpoint = tf.train.Checkpoint(model=model)
checkpoint.write('weights/vgg16')


