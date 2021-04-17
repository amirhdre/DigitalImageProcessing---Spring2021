'''
___OVERVIEW___
CODE FOR QUESTION 5 - HW 0

___SUMMARY___
Excercise with Image: Loading, modifying, and calculating its size and shape  

___AUTHOR___
Amirhossein Daraie — 9733023

___DATE___
2/28/2021

'''

import cv2 
import matplotlib.pyplot as plt
import numpy as np

# load image
im_gray = cv2.imread('fingerprint.tif', cv2.IMREAD_GRAYSCALE)

# flip image
flipVertical = cv2.flip(im_gray, 0)
flipHorizontal = cv2.flip(im_gray, 1)
flipBoth = cv2.flip(im_gray, -1)

# show images
fig,ax = plt.subplots(2,2)
fig.suptitle('HW0-P5-9733023-Fingerprint')
ax[0,0].imshow(cv2.cvtColor(im_gray, cv2.COLOR_BGR2RGB))
ax[0,0].set(title='Original Image')
ax[0,1].imshow(cv2.cvtColor(flipVertical, cv2.COLOR_BGR2RGB))
ax[0,1].set(title='Vertically Flipped Image')
ax[1,0].imshow(cv2.cvtColor(flipHorizontal, cv2.COLOR_BGR2RGB))
ax[1,0].set(title='Horizontally Flipped Image')
ax[1,1].imshow(cv2.cvtColor(flipBoth, cv2.COLOR_BGR2RGB))
ax[1,1].set(title='Both Axis Flipped Image')
for obj in ax.flatten():
    obj.axis('off')
    # print(type(obj),obj)
plt.tight_layout()
plt.show()

# we have 256 values on x axis, we want to plot histogram for each 8 of them
# so we will need 32 bins in our histogram
y,x = np.histogram(im_gray.flatten(),bins=32)
x = (x[1:]+x[:-1])/2
# calculate histogram and plot
barWidth = 8*0.5
fig,axes = plt.subplots(figsize =(12, 8))
axes.bar(x, y, width = barWidth)
plt.show()

# information about image 
y = im_gray.shape[0]
x = im_gray.shape[1]
total = x*y
UINT8_T = 1 # BYTES
pixel_to_byte = lambda p: p*UINT8_T
byte_to_kib = lambda b: b/1024

# print desired outputs
print()
print(f'height(y) is: {y}')
print(f'width(x) is: {x}')
print(f'Total pixels: {total}')
print(f'Pixel type: {type(im_gray[0,0])}')
print(f'Total size: {byte_to_kib(pixel_to_byte(total))} kibibyte')
print()