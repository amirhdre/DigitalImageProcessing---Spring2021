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

######################################
#### Bit plane slicing
######################################

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


######################################
#### Motion detection
######################################

img1 = cv2.imread("NASA-A.tif", flags=cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("NASA-B.tif", flags=cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread("NASA-C.tif", flags=cv2.IMREAD_GRAYSCALE)
result = []

### compare img1 with img2 && also img2 with img3

# compate img1 with img2
bps1_7 = bitplane_slice(img1, 7)
bps1_6 = bitplane_slice(img1, 6)
bps1_5 = bitplane_slice(img1, 5)
bps1_4 = bitplane_slice(img1, 4)

bps2_7 = bitplane_slice(img2, 7)
bps2_6 = bitplane_slice(img2, 6)
bps2_5 = bitplane_slice(img2, 5)
bps2_4 = bitplane_slice(img2, 4)

res_7 = cv2.bitwise_xor(bps1_7,bps2_7)
res_6 = cv2.bitwise_xor(bps1_6,bps2_6)
res_5 = cv2.bitwise_xor(bps1_5,bps2_5)
res_4 = cv2.bitwise_xor(bps1_4,bps2_4)

result.append(res_7+res_6+res_5+res_4)


# compate img2 with img3
bps1_7 = bitplane_slice(img2, 7)
bps1_6 = bitplane_slice(img2, 6)
bps1_5 = bitplane_slice(img2, 5)
bps1_4 = bitplane_slice(img2, 4)

bps2_7 = bitplane_slice(img3, 7)
bps2_6 = bitplane_slice(img3, 6)
bps2_5 = bitplane_slice(img3, 5)
bps2_4 = bitplane_slice(img3, 4)

res_7 = cv2.bitwise_xor(bps1_7,bps2_7)
res_6 = cv2.bitwise_xor(bps1_6,bps2_6)
res_5 = cv2.bitwise_xor(bps1_5,bps2_5)
res_4 = cv2.bitwise_xor(bps1_4,bps2_4)

result.append(res_7+res_6+res_5+res_4)

# plot results 
fig, ax = plt.subplots(1,2,figsize=(10,6))
st = fig.suptitle("Motion Detection", fontsize="x-large")
for i,axi in enumerate(ax.ravel()):
    outfinal_i = result[i]
    fig = axi.imshow(outfinal_i, vmin=outfinal_i.min(), vmax=outfinal_i.max(), cmap='gray')
    axi.title.set_text(f'Image {i+1},{i+2}')
    axi.axis('off')
    fig.axes.get_xaxis().set_visible(False),fig.axes.get_yaxis().set_visible(False)
plt.tight_layout()
# plt.savefig('9733023-4_image2.png', bbox_inches='tight')
plt.show()