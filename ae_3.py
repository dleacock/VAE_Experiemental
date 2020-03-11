import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import MaxPool2D, Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Activation, \
    LeakyReLU
from utils.loaders import load_minst

(X_train_full, y_train_full), (X_test, y_test) = load_minst()
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

encoder_input = Input(shape=(28, 28, 1), name='encoder_input')
x = encoder_input
x = Conv2D(filters=16, kernel_size=3, padding='same', name='encoder_conv_1', activation='selu')(x)
x = MaxPool2D(pool_size=2)(x)
x = Conv2D(filters=32, kernel_size=3, padding='same', name='encoder_conv_2', activation='selu')(x)
x = MaxPool2D(pool_size=2)(x)
x = Conv2D(filters=64, kernel_size=3, padding='same', name='encoder_conv_3', activation='selu')(x)
x = MaxPool2D(pool_size=2)(x)
x = Conv2D(filters=64, kernel_size=3, padding='same', name='encoder_conv_4', activation='selu')(x)
x = MaxPool2D(pool_size=2)(x)

shape_before_flattening = K.int_shape(x)[1:]
x = Flatten()(x)

encoder_output = Dense(2, name='encoder_output')(x)
encoder = tf.keras.models.Model(inputs=encoder_input, outputs=encoder_output)

decoder_input = Input(shape=(2,), name='decoder_input')
x = Dense(np.prod(shape_before_flattening))(decoder_input)
x = Reshape(shape_before_flattening)(x)

x = Conv2DTranspose(activation='selu', filters=32, kernel_size=3, strides=2, padding='valid', name='decoder_conv_1')(x)
x = Conv2DTranspose(activation='selu', filters=32, kernel_size=3, strides=2, padding='valid', name='decoder_conv_2')(x)
x = Conv2DTranspose(activation='selu', filters=16, kernel_size=3, strides=2, padding='same', name='decoder_conv_3')(x)
x = Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding='same', name='decoder_conv_4', activation='sigmoid')(x)
decoder_output = x
decoder = tf.keras.models.Model(decoder_input, decoder_output)

encoder.summary()
decoder.summary()

model_input = encoder_input
model_output = decoder(encoder_output)
model = tf.keras.models.Model(model_input, model_output)
model.summary()

optimizer = tf.keras.optimizers.Adam(lr=0.0005)


def r_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])


def rounded_accuracy(y_true, y_pred):
    return tf.keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))


# model.compile(optimizer=optimizer, loss=r_loss)
# model.fit(X_train, X_train, batch_size=32, epochs=200, initial_epoch=0, shuffle=True,
#           validation_data=[X_valid, X_valid])

model.compile(loss=r_loss, optimizer=optimizer, metrics=[rounded_accuracy])
model.fit(X_train, X_train, epochs=100, batch_size=32, validation_data=[X_valid, X_valid], initial_epoch=0, shuffle=True)

model.save('weights/autoencoder_1.h5')
decoder.save('weights/decoder_1.h5')
encoder.save('weights/encoder_1.h5')
