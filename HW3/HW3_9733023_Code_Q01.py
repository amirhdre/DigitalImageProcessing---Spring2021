'''
___OVERVIEW___
CODE FOR QUESTION 1 - HW 3

___SUMMARY___
Smoothing and Sharpening

___AUTHOR___
Amirhossein Daraie — 9733023

___DATE___
17/April/2021
28/01/1400

'''

import numpy as np
import matplotlib.pyplot as plt


# a dead-simple implementation of 2d-convolution with for loop
def convolution2d(image, kernel, stride=1, padding=0):
    image = np.pad(image, [(padding, padding), (padding, padding)], mode='constant', constant_values=0)

    kernel_height, kernel_width = kernel.shape
    padded_height, padded_width = image.shape

    output_height = (padded_height - kernel_height) // stride + 1
    output_width = (padded_width - kernel_width) // stride + 1

    new_image = np.zeros((output_height, output_width)).astype(np.float32)

    for y in range(0, output_height):
        for x in range(0, output_width):
            new_image[y][x] = np.sum(image[y * stride:y * stride + kernel_height, x * stride:x * stride + kernel_width] * kernel).astype(np.float32)
    return new_image



m = np.array([[0,0,10,10,0],
              [0,10,20,20,10],
              [0,10,20,20,10],
              [0,0,10,10,0],
              [0,0,0,0,0]])

smoothing1 = np.ones((3,3))/9
smoothed1 = convolution2d(m, smoothing1)

smoothing2 = np.array([[1,2,1],
                       [2,4,2],
                       [1,2,1]])/16
smoothed2 = convolution2d(m, smoothing2)

# laplacian mask
sharpening1 = np.array([[0,1,0],
                        [1,-4,1],
                        [0,1,0]]) 
sharped1 = convolution2d(m, sharpening1)

# Unsharp mask 
original = np.array([[0,0,0],
                        [0,1,0],
                        [0,0,0]])
detail = original - np.ones((3,3))/9
sharpening2 = original + detail
sharped2 = convolution2d(m, sharpening2)


# fig, ax = plt.subplots(figsize=(8,8))
# ax.title.set_text('Original Matrix')
# ax.imshow(m)
# for (j, i), val in np.ndenumerate(m):
#     label = f'{val:.2f}'
#     ax.text(i,j,label,ha='center',va='center',fontsize=9)
# ax.axis('off')
# plt.show()


fig, ax = plt.subplots(4,2,figsize=(4,10))
    
ax[0,0].title.set_text('Smoothing Kernel 1')
ax[0,0].imshow(smoothing1,interpolation='nearest')
for (j, i), val in np.ndenumerate(smoothing1):
    label = f'{val:.1f}'
    ax[0,0].text(i,j,label,ha='center',va='center',fontsize=9)


ax[0,1].title.set_text('Smoothed with Kernel 1')
ax[0,1].imshow(smoothed1,interpolation='nearest')
for (j, i), val in np.ndenumerate(smoothed1):
    label = f'{val:.1f}'
    ax[0,1].text(i,j,label,ha='center',va='center',fontsize=9)

ax[1,0].title.set_text('Smoothing Kernel 2')
ax[1,0].imshow(smoothing2,interpolation='nearest')
for (j, i), val in np.ndenumerate(smoothing2):
    label = f'{val:.1f}'
    ax[1,0].text(i,j,label,ha='center',va='center',fontsize=9)

ax[1,1].title.set_text('Smoothed with Kernel 2')
ax[1,1].imshow(smoothed2,interpolation='nearest')
for (j, i), val in np.ndenumerate(smoothed2):
    label = f'{val:.1f}'
    ax[1,1].text(i,j,label,ha='center',va='center',fontsize=9)

ax[2,0].title.set_text('Sharpening Kernel 1')
ax[2,0].imshow(sharpening1,interpolation='nearest')
for (j, i), val in np.ndenumerate(sharpening1):
    label = f'{val:.1f}'
    ax[2,0].text(i,j,label,ha='center',va='center',fontsize=9)

ax[2,1].title.set_text('Sharped with Kernel 1')
ax[2,1].imshow(sharped1,interpolation='nearest')
for (j, i), val in np.ndenumerate(sharped1):
    label = f'{val:.1f}'
    ax[2,1].text(i,j,label,ha='center',va='center',fontsize=9)

ax[3,0].title.set_text('Sharpening Kernel 2')
ax[3,0].imshow(sharpening2,interpolation='nearest')
for (j, i), val in np.ndenumerate(sharpening2):
    label = f'{val:.1f}'
    ax[3,0].text(i,j,label,ha='center',va='center',fontsize=9)

ax[3,1].title.set_text('Sharped with Kernel 2')
ax[3,1].imshow(sharped2,interpolation='nearest')
for (j, i), val in np.ndenumerate(sharped2):
    label = f'{val:.1f}'
    ax[3,1].text(i,j,label,ha='center',va='center',fontsize=9)

for axis in ax.ravel():
    axis.axis('off')

plt.tight_layout()
plt.show()


