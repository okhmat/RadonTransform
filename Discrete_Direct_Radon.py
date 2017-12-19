from skimage.io import imread
from skimage.transform import resize, rotate
import matplotlib.pyplot as plt
import numpy as np

N = 1024
num_images = 2

#======================================================================================================================
# Testing images loading/showing


plt.figure(figsize=(12, 6))

list_of_images = list()

for k in range(num_images):
    plt.subplot(1, 2, k+1)
    if k == 0:
        # rescale to the interval [0,1]
        list_of_images.append(imread('Shepp_logan.png', as_grey=True) / 255.0)
    elif k == 1:
        list_of_images.append(np.zeros((N, N)))
        # not necessary to resize in this case
        size = 1024
        # resize to (size * size)
        im = resize(imread('lena.gif', as_grey=True) / 255.0, (size, size))
        list_of_images[-1][(N - size) // 2:(N - size) // 2 + size,
        (N - size) // 2:(N - size) // 2 + size] = im

    # check, that
    assert list_of_images[-1].shape == (N, N)

    plt.imshow(list_of_images[-1], cmap=plt.cm.Greys_r)

plt.show()


#======================================================================================================================
# Test rotations


# then, we need to rotate the objects to get projections from different angles

plt.figure(figsize=(12,6))

for k in range(num_images):
    plt.subplot(1, 2, k+1)
    if k == 0:
        # rotate 'phantom' image on 60 degrees clockwise (-60 degrees)
        rotated_im = rotate(list_of_images[k], -60)
        assert rotated_im.shape == (N, N)
        plt.imshow(rotated_im, cmap=plt.cm.Greys_r)
    elif k == 1:
        # rotate 'Lena' on +30 degrees
        rotated_im = rotate(list_of_images[k], +30)
        assert rotated_im.shape == (N, N)
        plt.imshow(rotated_im, cmap=plt.cm.Greys_r)

plt.show()






