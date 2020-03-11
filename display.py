import tensorflow as tf
from utils.loaders import load_minst
import numpy as np
import matplotlib.pyplot as plt

def rounded_accuracy(y_true, y_pred):
    return tf.keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))

encoder = tf.keras.models.load_model('weights/encoder_1.h5', custom_objects={'rounded_accuracy': tf.keras.metrics.binary_accuracy})
decoder = tf.keras.models.load_model('weights/decoder_1.h5', custom_objects={'rounded_accuracy': tf.keras.metrics.binary_accuracy})

(x_train, y_train), (x_test, y_test) = load_minst()





def plot10():
    n_to_show = 10
    example_idx = np.random.choice(range(len(x_test)), n_to_show)
    example_images = x_test[example_idx]

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

    example_idx = np.random.choice(range(len(x_test)), n_to_show)
    example_images = x_test[example_idx]
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
    example_idx = np.random.choice(range(len(x_test)), n_to_show)
    example_images = x_test[example_idx]
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

    example_idx = np.random.choice(range(len(x_test)), n_to_show)
    example_images = x_test[example_idx]
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
