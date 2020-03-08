from utils.loaders import load_minst
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, \
    BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = load_minst()

encoder_input = Input(shape=(28, 28, 1), name='encoder_input')

x = encoder_input
x = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', name='encoder_conv_1')(x)
x = LeakyReLU()(x)
x = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', name='encoder_conv_2')(x)
x = LeakyReLU()(x)
x = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', name='encoder_conv_3')(x)
x = LeakyReLU()(x)
x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', name='encoder_conv_4')(x)
x = LeakyReLU()(x)

shape_before_flattening = K.int_shape(x)[1:]

x = Flatten()(x)

encoder_output = Dense(2, name='encoder_output')(x)
encoder = tf.keras.models.Model(inputs=encoder_input, outputs=encoder_output)

decoder_input = Input(shape=(2,), name='decoder_input')
x = Dense(np.prod(shape_before_flattening))(decoder_input)
x = Reshape(shape_before_flattening)(x)

x = Conv2DTranspose(filters=64, kernel_size=3, strides=1, padding='same', name='decoder_conv_1')(x)
x = LeakyReLU()(x)
x = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', name='decoder_conv_2')(x)
x = LeakyReLU()(x)
x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', name='decoder_conv_3')(x)
x = LeakyReLU()(x)
x = Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same', name='decoder_conv_4')(x)
x = LeakyReLU()(x)
x = Activation('sigmoid')(x)
decoder_output = x
decoder = tf.keras.models.Model(decoder_input, decoder_output)

model_input = encoder_input
model_output = decoder(encoder_output)
model = tf.keras.models.Model(model_input, model_output)


model.summary()

model.save('weights/autoencoder.h5')
optimizer = tf.keras.optimizers.Adam(lr=0.0005)


def r_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])


def loss(y_true, y_pred):
    return K.sum(K.binary_crossentropy(y_pred, y_true), axis=[1, 2, 3])


#model.compile(optimizer=optimizer, loss=r_loss)
#model.fit(x_train[:1000], x_train[:1000], batch_size=32, epochs=200, initial_epoch=0, shuffle=True)
#model.save('weights/autoencoder_1.h5')
model.load_weights('weights/autoencoder_1.h5')

n_to_show = 10
example_idx = np.random.choice(range(len(x_test)), n_to_show)
example_images = x_test[example_idx]

z_points = encoder.predict(example_images)

reconst_images = decoder.predict(z_points)

fig = plt.figure(figsize=(15, 3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i in range(n_to_show):
    img = example_images[i].squeeze()
    ax = fig.add_subplot(2, n_to_show, i+1)
    ax.axis('off')
    ax.text(0.5, -0.35, str(np.round(z_points[i],1)), fontsize=10, ha='center', transform=ax.transAxes)
    ax.imshow(img, cmap='gray_r')

for i in range(n_to_show):
    img = reconst_images[i].squeeze()
    ax = fig.add_subplot(2, n_to_show, i+n_to_show+1)
    ax.axis('off')
    ax.imshow(img, cmap='gray_r')

plt.show()