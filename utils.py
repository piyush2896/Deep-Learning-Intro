import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def get_data_from_dir(path, size=[100, 100]):
    imgs = []
    labels = []
    labels_list = []

    files = os.listdir(path)

    for i, file in enumerate(files):
        img_names = os.listdir(os.path.join(path, file))
        
        labels.append(np.ones(len(img_names)) * i)
        labels_list.append(file)

        for img_name in img_names:
            with Image.open(os.path.join(path, file, img_name)) as img:
                imgs.append(np.asarray(img.resize(size)))

    imgs = np.asarray(imgs)
    print('Found {} images belonging to {} different classes'.format(imgs.shape[0], len(labels_list)))

    return imgs, np.hstack(labels), labels_list

def train_test_split(X, y, val_split=0.2):
    index = int(X.shape[0] * (1 - val_split))
    return X[:index], y[:index], X[index:], y[index:]

def shuffle(X, y):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    return X[indices], y[indices]

def one_hot(y):
    return np.eye(int(np.max(y)) + 1)[y.astype('int')]

def plot_some(X, y, y_hat=None, labels_list=None):
    indices = np.random.randint(0, X.shape[0], size=10)
    print(indices)

    if labels_list is None:
        labels_list = np.sort(np.unique(y))

    plt.figure(figsize=(15, 5))

    for i, index in enumerate(indices):
        plt.subplot(2, 5, i+1)
        plt.imshow(X[index])
        plt.title('Y: {}    Y_hat: {}'. \
                  format(labels_list[int(y[index])], 
                         'N/A' if y_hat is None else labels_list[y_hat[index]]))
        plt.axis('off')

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()
