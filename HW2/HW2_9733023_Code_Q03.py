'''
___OVERVIEW___
CODE FOR QUESTION 3 - HW 2

___SUMMARY___
Transfer Functons

___AUTHOR___
Amirhossein Daraie — 9733023

___DATE___
08/April/2021
19/01/1400

'''

import numpy as np
import cv2
import matplotlib.pyplot as plt

#####################################
## Part 1
#####################################


def s(r):
    L = 4 #intesities from 0,1,2,...,3
    a =  1/L
    return a*(r**2+r)

arr = np.array([[1,2,3],
                [3,2,1],
                [1,2,2]])

print( s(arr) )


#####################################
## Part 2
#####################################

def transform(img, bd): 
    """Apply s(r)=1/bd*(r**2+r) transformation on input \n
    Example: \n
    For example if bd=2, then L=4, and intensities can be 0,1,2,3\n
    For example if bd=3, then L=8, and intensities can be 0,1,2,3,4,5,6,7\n
    For example if bd=8, then L=256, and intensities can be 0,1,2,...,255\n
    
    Args:
        img (np.array): Input image/array
        bd (int): Number of bits for input image/array 

    Returns:
        np.array: Transformed Input, with same data type as input 
    """
    print(type(np.ravel(img)[0]))
    
    L = 2**bd # number of bits 
    # for example if bd=2, then L=4, and intensities can be 0,1,2,3
    # for example if bd=3, then L=8, and intensities can be 0,1,2,3,4,5,6,7
    # for example if bd=8, then L=256, and intensities can be 0,1,2,...,255
    a = 1/L
    
    result = img.astype(np.float32)
    result = a*(result**2 + result)
    print(type(np.ravel(result)[0]))
    
    result = np.round(result)
    result = result.astype(type(np.ravel(img)[0]))
    print(type(np.ravel(result)[0]))
    
    return result
 
 
# example for transform function
arr = np.array([[1,2,3],
                [3,2,1],
                [1,2,2]])
print('Input: ')
print(arr)
print('Transformed Output: ')
print( transform(arr,4) )

# load image as color image
image = cv2.imread("kidney.tif", flags=cv2.IMREAD_UNCHANGED)
# apply transformation on image
transformed_image = transform(image,8)

# load image as color image
image2 = cv2.imread("chest.tif", flags=cv2.IMREAD_UNCHANGED)
# apply transformation on image
transformed_image2 = transform(image2,16)

# show images 
fig, ax = plt.subplots(2,2,figsize=(6,6))
st = fig.suptitle("Apply Transformation on 8 bit Image", fontsize="x-large")
ax[0,0].imshow(image,vmin=0,vmax=2**8-1)
ax[0,0].set_title('8 bit Image')
ax[0,1].imshow(transformed_image,vmin=0,vmax=2**8-1)
ax[0,1].set_title('Transformed 8 bit Image')
ax[1,0].hist(image,bins=(2**8//4))
ax[1,0].set_title('Histogram')
ax[1,1].hist(transformed_image,bins=(2**8//4))
ax[1,1].set_title('Histogram')
ax[0,0].axis('off')
ax[0,1].axis('off')
plt.tight_layout()
# plt.savefig('9733023-3_image1.png', bbox_inches='tight')
plt.show()

# show images
fig, ax = plt.subplots(2,2,figsize=(6,6))
st = fig.suptitle("Apply Transformation on 16 bit Image", fontsize="x-large")
ax[0,0].imshow(image2,vmin=0,vmax=2**16-1)
ax[0,0].set_title('16 bit Image')
ax[0,1].imshow(transformed_image2,vmin=0,vmax=2**16-1)
ax[0,1].set_title('Transformed 16 bit Image')
ax[1,0].hist(image2,bins=(2**16//1024))
ax[1,0].set_title('Histogram')
ax[1,1].hist(transformed_image2,bins=(2**16//1024))
ax[1,1].set_title('Histogram')
ax[0,0].axis('off')
ax[0,1].axis('off')
plt.tight_layout()
# plt.savefig('9733023-3_image2.png', bbox_inches='tight')
plt.show()




#####################################
## Part 3
#####################################

fig, ax = plt.subplots(figsize=(6,6))
st = fig.suptitle("Transfer Functons", fontsize="x-large")
nbits = 16
intensities = np.arange(0,2**nbits) # upper-bound excluded
ax.plot(intensities, intensities, 'b--', label='S(r)=r')
result = transform(intensities, nbits) 
ax.plot(intensities, result, 'k', label='$S(r)=a^{-1}*(r^2+r)$') 
ax.plot(intensities, np.log(intensities), 'g-', label='S(r)=log10(r)')
ax.legend()
plt.tight_layout()
# plt.savefig('9733023-3_image3.png', bbox_inches='tight')
plt.show()

print(np.log10(intensities))





