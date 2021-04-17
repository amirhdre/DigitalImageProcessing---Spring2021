'''
___OVERVIEW___
CODE FOR QUESTION 4 - HW 2

___SUMMARY___
Various Transfer Functions

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

def transform2(img, A, B):
    function = lambda x,A,B : x if (x >= A and x <= B) else 0
    
    arr = []
    for num in img.ravel():
        arr.append( function(num,A,B) )
    
    return np.array(arr).reshape(img.shape)
    
A = np.array([[1,2,3],
             [3,0,5],
             [4,3,2]])

print("Input:")
print(A)
print("Output:")
print(transform2(A,1,3))


#####################################
## Part 2
#####################################

# load and process images
image = cv2.imread("HeadCT.tif", flags=cv2.IMREAD_GRAYSCALE)
result = transform2(image,55,250)
# show images
fig,ax = plt.subplots(1,2,figsize=(12,6))
ax[0].imshow(image,vmin=image.min(),vmax=image.max(), cmap='gray')
ax[1].imshow(result,vmin=result.min(),vmax=result.max(), cmap='gray')
ax[0].set_title('Before')
ax[1].set_title('After')
ax[0].axis('off')
ax[1].axis('off')
plt.tight_layout()
# plt.savefig('9733023-4_image1.png', bbox_inches='tight')
plt.show()


#####################################
## Part 3
#####################################

intensities = np.arange(0,2**8-1)
fig,ax = plt.subplots(figsize=(6,6))
st = fig.suptitle("Transfer Functon", fontsize="x-large")
ax.plot(intensities,transform2(intensities,55,250),lw=2)
plt.tight_layout()
# plt.savefig('9733023-4_image2.png', bbox_inches='tight')
plt.show()

