'''
___OVERVIEW___
CODE FOR QUESTION 3 - HW 3

___SUMMARY___


___AUTHOR___
Amirhossein Daraie — 9733023

___DATE___
18/April/2021
29/01/1400

'''

import numpy as np
import matplotlib.pyplot as plt
import cv2


def filter(image ,mode="averaging", size=3, padding=1, stride=1):   
    
    image = np.pad(image, [(padding, padding), (padding, padding)], mode='symmetric')

    kernel_height, kernel_width = (size,size)
    padded_height, padded_width = image.shape

    output_height = (padded_height - kernel_height) // stride + 1
    output_width = (padded_width - kernel_width) // stride + 1

    new_image = np.zeros((output_height, output_width)).astype(np.float32)
        
        
    if mode == "averaging" :
        kernel = np.ones((size,size))/size**2
        for y in range(0, output_height):
            for x in range(0, output_width):
                new_image[y][x] = np.sum(image[y * stride:y * stride + kernel_height, 
                                                x * stride:x * stride + kernel_width] * kernel).astype(np.float32)
                
    elif mode == "minimum":
        kernel = np.ones((size,size))
        for y in range(0, output_height):
            for x in range(0, output_width):
                new_image[y][x] = np.min(image[y * stride:y * stride + kernel_height, 
                                                x * stride:x * stride + kernel_width]).astype(np.float32)
                
    elif mode == "median":
        kernel = np.ones((size,size))
        for y in range(0, output_height):
            for x in range(0, output_width):
                new_image[y][x] = np.median(image[y * stride:y * stride + kernel_height, 
                                                x * stride:x * stride + kernel_width]).astype(np.float32)
                
    elif mode == "sobel_y":
        kernel = np.array([[-1,-2,-1],
                            [ 0, 0, 0],
                            [ 1, 2, 1]]) 
        for y in range(0, output_height):
            for x in range(0, output_width):
                new_image[y][x] = np.sum(image[y * stride:y * stride + kernel_height, 
                                                x * stride:x * stride + kernel_width] * kernel).astype(np.float32)
                
    elif mode == "laplacian":
        kernel = np.array([[0,1,0 ],
                            [1,-4,1],
                            [0,1,0]]) 
        for y in range(0, output_height):
            for x in range(0, output_width):
                new_image[y][x] = np.sum(image[y * stride:y * stride + kernel_height, 
                                                x * stride:x * stride + kernel_width] * kernel).astype(np.float32)
                
    elif mode == "custom":
        kernel = np.array([[-1,-1,0 ],
                            [-1,0,1],
                            [0,1,1]]) 
        for y in range(0, output_height):
            for x in range(0, output_width):
                new_image[y][x] = np.sum(image[y * stride:y * stride + kernel_height, 
                                                x * stride:x * stride + kernel_width] * kernel).astype(np.float32)

    return new_image

    

# load image
image = cv2.imread("MRI.png", flags=cv2.IMREAD_GRAYSCALE)
# apply filters
avg3x3 = filter(image, mode='averaging')
min3x3 = filter(image, mode='minimum')
med3x3 = filter(image, mode='median')
sob = filter(image, mode='sobel_y')
lap = filter(image, mode='laplacian')
ctm = filter(image, mode='custom')
avg7x7 = filter(image, mode='averaging', size=7)
min7x7 = filter(image, mode='minimum', size=7)
med7x7 = filter(image, mode='median', size=7)

# plot results
fig, ax = plt.subplots(3,3,figsize=(10,10))
st = fig.suptitle("Filter Image (linear and non-linear)", fontsize="x-large")

ax[0,0].title.set_text('averaging (3x3)')
ax[0,0].imshow(avg3x3,vmin=avg3x3.min(),vmax=avg3x3.max(), cmap='gray')

ax[0,1].title.set_text('minimum (3x3)')
ax[0,1].imshow(min3x3,vmin=min3x3.min(),vmax=min3x3.max(), cmap='gray')

ax[0,2].title.set_text('median (3x3)')
ax[0,2].imshow(med3x3,vmin=med3x3.min(),vmax=med3x3.max(), cmap='gray')

ax[1,0].title.set_text('sobel_y')
ax[1,0].imshow(sob,vmin=sob.min(),vmax=sob.max(), cmap='gray')

ax[1,1].title.set_text('laplacian')
ax[1,1].imshow(lap,vmin=lap.min(),vmax=lap.max(), cmap='gray')

ax[1,2].title.set_text('custom')
ax[1,2].imshow(ctm,vmin=ctm.min(),vmax=ctm.max(), cmap='gray')

ax[2,0].title.set_text('averaging (7x7)')
ax[2,0].imshow(avg7x7,vmin=avg7x7.min(),vmax=avg7x7.max(), cmap='gray')

ax[2,1].title.set_text('minimum (7x7)')
ax[2,1].imshow(min7x7,vmin=min7x7.min(),vmax=min7x7.max(), cmap='gray')

ax[2,2].title.set_text('median (7x7)')
ax[2,2].imshow(med7x7,vmin=med7x7.min(),vmax=med7x7.max(), cmap='gray')

for axi in ax.ravel():
    axi.axis('off')
plt.tight_layout()
# plt.savefig('9733023-2_image1.png', bbox_inches='tight')
plt.show()
