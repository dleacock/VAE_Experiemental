import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Activation, LeakyReLU, \
    MaxPool2D
from tensorflow.keras.models import Sequential
from utils.loaders import load_minst
import matplotlib.pyplot as plt

tf.random.set_seed(42)
np.random.seed(42)

# X_train_full = X_train_full.astype(np.float32) / 255
# X_test = X_test.astype(np.float32) / 255

(X_train_full, y_train_full), (X_test, y_test) = load_minst()
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

encoder = Sequential([
    Reshape([28, 28, 1], input_shape=[28, 28, 1]),
    Conv2D(16, kernel_size=3, padding='same', activation='selu'),
    MaxPool2D(pool_size=2),
    Conv2D(32, kernel_size=3, padding='same', activation='selu'),
    MaxPool2D(pool_size=2),
    Conv2D(64, kernel_size=3, padding='same', activation='selu'),
    MaxPool2D(pool_size=2)
])

decoder = Sequential([
    Conv2DTranspose(32, kernel_size=3, strides=2, padding='valid', activation='selu', input_shape=[3, 3, 64]),
    Conv2DTranspose(16, kernel_size=3, strides=2, padding='same', activation='selu'),
    Conv2DTranspose(1, kernel_size=3, strides=2, padding='same', activation='sigmoid'),
    Reshape([28, 28])
])


def rounded_accuracy(y_true, y_pred):
    return tf.keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))


autoencoder = Sequential([encoder, decoder])

encoder.summary()
decoder.summary()

autoencoder.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=1.0), metrics=[rounded_accuracy])
autoencoder.fit(X_train, X_train, epochs=5, validation_data=[X_valid, X_valid], initial_epoch=0, shuffle=True)

autoencoder.save("weights/autoencoder_m2.h5")
encoder.save("weights/encoder_m2.h5")
decoder.save("weights/decoder_m2.h5")



def plot10():
    n_to_show = 10
    example_idx = np.random.choice(range(len(X_test)), n_to_show)
    example_images = X_test[example_idx]

    z_points = encoder.predict(example_images)
    reconst_images = decoder.predict(z_points)

    fig = plt.figure(figsize=(15, 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(n_to_show):
        img = example_images[i].squeeze()
        ax = fig.add_subplot(2, n_to_show, i + 1)
        ax.axis('off')
        ax.text(0.5, -0.35, str(np.round(z_points[i], 1)), fontsize=10, ha='center', transform=ax.transAxes)
        ax.imshow(img, cmap='gray_r')

    for i in range(n_to_show):
        img = reconst_images[i].squeeze()
        ax = fig.add_subplot(2, n_to_show, i + n_to_show + 1)
        ax.axis('off')
        ax.imshow(img, cmap='gray_r')
    plt.show()


def wall():
    n_to_show = 5000
    grid_size = 15
    figsize = 12

    example_idx = np.random.choice(range(len(X_test)), n_to_show)
    example_images = X_test[example_idx]
    example_labels = y_test[example_idx]

    z_points = encoder.predict(example_images)

    min_x = min(z_points[:, 0])
    max_x = max(z_points[:, 0])
    min_y = min(z_points[:, 1])
    max_y = max(z_points[:, 1])

    plt.figure(figsize=(figsize, figsize))
    plt.scatter(z_points[:, 0], z_points[:, 1], c='black', alpha=0.5, s=2)
    plt.show()


def samplewall():
    n_to_show = 5000
    figsize = 5
    example_idx = np.random.choice(range(len(X_test)), n_to_show)
    example_images = X_test[example_idx]
    example_labels = y_test[example_idx]

    z_points = encoder.predict(example_images)

    min_x = min(z_points[:, 0])
    max_x = max(z_points[:, 0])
    min_y = min(z_points[:, 1])
    max_y = max(z_points[:, 1])

    plt.figure(figsize=(figsize, figsize))
    plt.scatter(z_points[:, 0], z_points[:, 1], c='black', alpha=0.5, s=2)

    grid_size = 10
    grid_depth = 3
    figsize = 15

    x = np.random.uniform(min_x, max_x, size=grid_size * grid_depth)
    y = np.random.uniform(min_y, max_y, size=grid_size * grid_depth)
    z_grid = np.array(list(zip(x, y)))
    reconst = decoder.predict(z_grid)

    plt.scatter(z_grid[:, 0], z_grid[:, 1], c='red', alpha=1, s=20)
    plt.show()

    fig = plt.figure(figsize=(figsize, grid_depth))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(grid_size * grid_depth):
        ax = fig.add_subplot(grid_depth, grid_size, i + 1)
        ax.axis('off')
        ax.text(0.5, -0.35, str(np.round(z_grid[i], 1)), fontsize=10, ha='center', transform=ax.transAxes)

        ax.imshow(reconst[i, :, :, 0], cmap='Greys')


def b():
    n_to_show = 5000
    grid_size = 15
    figsize = 12

    example_idx = np.random.choice(range(len(X_test)), n_to_show)
    example_images = X_test[example_idx]
    example_labels = y_test[example_idx]

    z_points = encoder.predict(example_images)

    plt.figure(figsize=(figsize, figsize))
    plt.scatter(z_points[:, 0], z_points[:, 1], cmap='rainbow', c=example_labels
                , alpha=0.5, s=2)
    plt.colorbar()
    plt.show()


plot10()
wall()
b()
