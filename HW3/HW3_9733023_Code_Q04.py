'''
___OVERVIEW___
CODE FOR QUESTION 4 - HW 3

___SUMMARY___
Bit plane slicing
Motion detector


___AUTHOR___
Amirhossein Daraie — 9733023

___DATE___
20/April/2021
31/01/1400

'''

import numpy as np
import matplotlib.pyplot as plt
import cv2


def bitplane_slice(image,n):
    if n>7: n=7
    out = []
    # create an image for the n bit plane
    plane = np.full((image.shape[0], image.shape[1]), 2 ** n, np.uint8)
    # execute bitwise and operation
    res = cv2.bitwise_and(plane, image)
    # multiply ones (bit plane sliced) with 255 just for better visualization
    x = res * 255
    return x
    
img = cv2.imread("PCB.tif", flags=cv2.IMREAD_GRAYSCALE)
outfinal = bitplane_slice(img, 7)
# fig = plt.imshow(outfinal, vmin=outfinal.min(), vmax=outfinal.max(), cmap='gray')
# plt.axis('off')
# fig.axes.get_xaxis().set_visible(False),fig.axes.get_yaxis().set_visible(False)
# plt.show()

fig, ax = plt.subplots(2,4,figsize=(15,8))
st = fig.suptitle("Bit plane slicing", fontsize="x-large")
for i,axi in enumerate(ax.ravel()):
    outfinal_i =  bitplane_slice(img, i)
    fig = axi.imshow(outfinal_i, vmin=outfinal_i.min(), vmax=outfinal_i.max(), cmap='gray')
    axi.title.set_text(f'bit {i}')
    axi.axis('off')
    fig.axes.get_xaxis().set_visible(False),fig.axes.get_yaxis().set_visible(False)
plt.tight_layout()
# plt.savefig('9733023-4_image1.png', bbox_inches='tight')
plt.show()