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
#
for i in range(num_images):
    plt.subplot(1, 2, i+1)
    if i == 0:
        # rescale to the interval [0,1]
        list_of_images.append(imread('Shepp_logan.png', as_grey=True) / 255.0)
    elif i == 1:
        list_of_images.append(np.zeros((N, N)))
        # not necessary to resize in this case
        size = 1024
        # resize to (size * size)
        im = resize(imread('lena.gif', as_grey=True) / 255.0, (size, size))
        list_of_images[-1][ (N - size) // 2:(N - size) // 2 + size,
                            (N - size) // 2:(N - size) // 2 + size  ] = im

    # check, that the current downloading image has correct shape
    assert list_of_images[-1].shape == (N, N)

    plt.imshow(list_of_images[-1], cmap=plt.cm.Greys_r)
    if i == 0:
        plt.title('Original Phantom 1024x1024 pix.')
        plt.xlabel('x')
        plt.ylabel('y')
    elif i == 1:
        plt.title('Original Lena 1024x1024 pix.')
        plt.xlabel('x')
        plt.ylabel('y')

plt.show()


#======================================================================================================================
# Test rotations


# then, we need to rotate the objects to get projections from different angles

plt.figure(figsize=(12,6))

phantom_angle = -60
lena_angle = +30

for i in range(num_images):
    plt.subplot(1, 2, i+1)
    if i == 0:
        # rotate 'phantom' image on 60 degrees clockwise (-60 degrees)
        rotated_im = rotate(list_of_images[i], -60)
        assert rotated_im.shape == (N, N)

        plt.imshow(rotated_im, cmap=plt.cm.Greys_r)
        plt.title('Rotated on {} degrees Phantom'.format(phantom_angle))
        plt.xlabel('x')
        plt.ylabel('y')

    elif i == 1:
        # rotate 'Lena' on +30 degrees
        rotated_im = rotate(list_of_images[i], +30)
        assert rotated_im.shape == (N, N)

        plt.imshow(rotated_im, cmap=plt.cm.Greys_r)
        plt.title('Rotated on {} degrees Lena'.format(lena_angle))
        plt.xlabel('x')
        plt.ylabel('y')

plt.show()



#======================================================================================================================
# Radon transform


# observation angles discretization parameter - number of observation angles
# each angle = (180 degrees) / (num_of_angles)
num_of_angles = 90 # rotate on 2 degree each time


def radon(image, num_of_angles):

    angles = np.linspace(0, 180, num_of_angles)
    assert image.shape[0] == N # size of matrix dimension (1024)
    assert image.shape[0] == image.shape[1] # square matrix

    sinogram = np.zeros((N, len(angles)))

    for n, alpha in enumerate(angles):
        rotated_im = rotate(image, -alpha) # clockwise, but it doesn't matter
        sinogram[:, n] = np.sum(rotated_im, axis=0)

    return sinogram



#======================================================================================================================
# Plotting sinograms of classical examples "Lena" and "Phantom"



plt.figure(figsize=(12,6))

# list of sinograms for each picture (out of 2)
list_of_sinograms = list()

for i in range(num_images):
    # create set of subplots
    plt.subplot(1, 2, i+1)

    # rotated_im = rotate(list_of_images[k], -60)
    # plt.imshow(rotated_im, cmap=plt.cm.Greys_r)

    sinogram = radon(list_of_images[i], num_of_angles)
    list_of_sinograms.append(sinogram)

    plt.imshow( sinogram,
                cmap=plt.cm.Greys_r,
                aspect='auto', # If ‘auto’, changes the image aspect ratio to match that of the axes. (коэффициент сжатия)
                extent=[0, 180, (-N)/2, N/2]  # If extent is not None, the axes aspect ratio is changed to match that of the extent.
              )
    if i == 0:
        plt.title('Sinogram of Phantom for {} angles'.format(num_of_angles))
        plt.xlabel('Observation angle, [degrees]')
        plt.ylabel('Sinogram (attenuation distribution)')
    elif i == 1:
        plt.title('Sinogram of Lena for {} angles'.format(num_of_angles))
        plt.xlabel('Observation angle, [degrees]')
        plt.ylabel('Sinogram (attenuation distribution)')

plt.show()


#======================================================================================================================
# Inverse Radon transform (backprojection version - Dual Radon Transform)
# iradon_fft(sinogram) - Mozhde


# I didn't rescale image gradient of color here
def iradon_dual(sinogram):
    assert sinogram.shape[0] == N
    assert sinogram.shape[1] == num_of_angles

    x = np.arange(N)

    # the origin in the center of an image
    X, XT = np.meshgrid( (x - (N/2)), -(x - (N/2)) )

    image = np.zeros((N, N))
    # sinogram.shape[1] == num_of_angles  (90 for now)
    angles = np.linspace(0, np.pi, sinogram.shape[1], endpoint=False)

    for n, alpha in enumerate(angles):
        # rotate image to obtain the current projection
        slice = X * np.cos(alpha) + XT * np.sin(alpha)
        # let's reduce values of S+(1/2)*N to the interval [0, 1023]: if a > 1023 -> a:=1023; if a <= 0 -> a:=0
        slice = np.clip(a=slice+(N/2), a_min=0, a_max=N-1, out=None)
        # we will use 'slice' as indeces, so it should be consisted of integers
        slice = slice.astype(int)
        # reconstruct image slicewise
        image += sinogram[slice, n]
        print('observation angle (alpha [radian]): ', alpha)
        print('======================================================================================================')
        print('shape of sinogram[slice, n]: ', sinogram[slice, n].shape)
        print('sinogram[slice, n]: ')
        print('======================================================================================================')
        print(sinogram[slice, n])
        print('======================================================================================================')

    image = image / len(angles)
    assert image.shape == (N, N)
    return image



plt.figure(figsize=(12, 6))
list_of_recovered_im = list()

for i in range(num_images):
    plt.subplot(1, 2, i+1)
    recovered_im = iradon_dual(list_of_sinograms[i])
    list_of_recovered_im.append(recovered_im)

    plt.imshow(recovered_im, cmap=plt.cm.Greys_r)
    if i == 0:
        plt.title('Recovered Phantom (blurred)')
        plt.xlabel('x')
        plt.ylabel('y')
    elif i == 1:
        plt.title('Recovered Lena (blurred)')
        plt.xlabel('x')
        plt.ylabel('y')

plt.show()



#======================================================================================================================
# Debluring is coming soon (I'm gonna do it through the night)
