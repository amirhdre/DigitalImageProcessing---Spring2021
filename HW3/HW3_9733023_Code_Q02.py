'''
___OVERVIEW___
CODE FOR QUESTION 2 - HW 3

___SUMMARY___
Filterning (linear and non-linear)

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
     
    if mode == "averaging" :
        kernel = np.ones((size,size))/size**2
        image = np.pad(image, [(padding, padding), (padding, padding)], mode='symmetric')

        kernel_height, kernel_width = kernel.shape
        padded_height, padded_width = image.shape

        output_height = (padded_height - kernel_height) // stride + 1
        output_width = (padded_width - kernel_width) // stride + 1

        new_image = np.zeros((output_height, output_width)).astype(np.float32)

        for y in range(0, output_height):
            for x in range(0, output_width):
                new_image[y][x] = np.sum(image[y * stride:y * stride + kernel_height, 
                                                x * stride:x * stride + kernel_width] * kernel).astype(np.float32)
                
    elif mode == "minimum":
        kernel = np.ones((size,size))
        image = np.pad(image, [(padding, padding), (padding, padding)], mode='symmetric')

        kernel_height, kernel_width = kernel.shape
        padded_height, padded_width = image.shape

        output_height = (padded_height - kernel_height) // stride + 1
        output_width = (padded_width - kernel_width) // stride + 1

        new_image = np.zeros((output_height, output_width)).astype(np.float32)

        for y in range(0, output_height):
            for x in range(0, output_width):
                new_image[y][x] = np.min(image[y * stride:y * stride + kernel_height, 
                                                x * stride:x * stride + kernel_width]).astype(np.float32)
                
    elif mode == "median":
        kernel = np.ones((size,size))
        image = np.pad(image, [(padding, padding), (padding, padding)], mode='symmetric')

        kernel_height, kernel_width = kernel.shape
        padded_height, padded_width = image.shape

        output_height = (padded_height - kernel_height) // stride + 1
        output_width = (padded_width - kernel_width) // stride + 1

        new_image = np.zeros((output_height, output_width)).astype(np.float32)

        for y in range(0, output_height):
            for x in range(0, output_width):
                new_image[y][x] = np.median(image[y * stride:y * stride + kernel_height, 
                                                x * stride:x * stride + kernel_width]).astype(np.float32)
                
    elif mode == "sobel_y":
        kernel = np.array([[-1,-2,-1],
                            [ 0, 0, 0],
                            [ 1, 2, 1]]) 
        image = np.pad(image, [(padding, padding), (padding, padding)], mode='symmetric')

        kernel_height, kernel_width = kernel.shape
        padded_height, padded_width = image.shape

        output_height = (padded_height - kernel_height) // stride + 1
        output_width = (padded_width - kernel_width) // stride + 1

        new_image = np.zeros((output_height, output_width)).astype(np.float32)

        for y in range(0, output_height):
            for x in range(0, output_width):
                new_image[y][x] = np.sum(image[y * stride:y * stride + kernel_height, 
                                                x * stride:x * stride + kernel_width] * kernel).astype(np.float32)
                
    elif mode == "laplacian":
        kernel = np.array([[0,1,0 ],
                            [1,-4,1],
                            [0,1,0]]) 
        image = np.pad(image, [(padding, padding), (padding, padding)], mode='symmetric')

        kernel_height, kernel_width = kernel.shape
        padded_height, padded_width = image.shape

        output_height = (padded_height - kernel_height) // stride + 1
        output_width = (padded_width - kernel_width) // stride + 1

        new_image = np.zeros((output_height, output_width)).astype(np.float32)
    
        for y in range(0, output_height):
            for x in range(0, output_width):
                new_image[y][x] = np.sum(image[y * stride:y * stride + kernel_height, 
                                                x * stride:x * stride + kernel_width] * kernel).astype(np.float32)
                
    elif mode == "custom":
        kernel = np.array([[-1,-1,0 ],
                            [-1,0,1],
                            [0,1,1]]) 
        image = np.pad(image, [(padding, padding), (padding, padding)], mode='symmetric')

        kernel_height, kernel_width = kernel.shape
        padded_height, padded_width = image.shape

        output_height = (padded_height - kernel_height) // stride + 1
        output_width = (padded_width - kernel_width) // stride + 1

        new_image = np.zeros((output_height, output_width)).astype(np.float32)
    
        for y in range(0, output_height):
            for x in range(0, output_width):
                new_image[y][x] = np.sum(image[y * stride:y * stride + kernel_height, 
                                                x * stride:x * stride + kernel_width] * kernel).astype(np.float32)

    return new_image

    
# load image
image = cv2.imread("MRI.png", flags=cv2.IMREAD_GRAYSCALE)
result = filter(image, mode='custom', size=3, padding=1, stride=1)
plt.imshow(result,vmin=result.min(),vmax=result.max(), cmap='gray')
plt.show()
