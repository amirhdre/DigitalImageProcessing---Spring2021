'''
___OVERVIEW___
CODE FOR QUESTION 2 - HW 1

___SUMMARY___
Study data in a photo (numpy array)

___AUTHOR___
Amirhossein Daraie — 9733023

___DATE___
24/03/2021
04/01/1400

'''

import numpy as np
import cv2
import matplotlib.pyplot as plt

#####################################
## Part 1
#####################################

# load image
image = cv2.imread("Head-MRI.tif", flags=cv2.IMREAD_GRAYSCALE)
fig, ax = plt.subplots(2,1,figsize=(6,10))
# plot uint8 image
ax[0].imshow(image,cmap='gray', vmin=0, vmax=255)
ax[0].set_title('Uint8 Image of Head-MRI')
# convert image to float32
image = image.astype(np.float32)
image = image / 255.0
# plot float32 image
ax[1].imshow(image,cmap='gray', vmin=0, vmax=1)
ax[1].set_title('Float32 Image of Head-MRI')
plt.show()


#####################################
## Part 2,3
#####################################

row150 = image[149,:]
row180 = image[179,:]

fig,ax = plt.subplots(2,1,figsize=(6,7))
st = fig.suptitle("Details of Row 150 and 180 of Head-MRI Image", fontsize="x-large")
# plot intensities 
ax[0].plot(row150,'b',label='Row 150')
ax[0].plot(row180,'r',label='Row 180')
ax[0].set_title('Intesity of row 150 and 160')
ax[0].legend()
# Add batch dimension for plotting.
row150 = np.expand_dims(row150, 0)  # Add batch dimension for plotting.
row180 = np.expand_dims(row180, 0)  # Add batch dimension for plotting.
# stack two rows to show in same subplot
row150_and_180 = np.vstack((row150, row180)) 
ax[1].imshow(row150_and_180,cmap='gray', vmin=0, vmax=1)
ax[1].set_title('Row 150 and 180 of Head-MRI Image')
ax[1].set_axis_off()
# shift subplots down:
st.set_y(0.95)
fig.subplots_adjust(top=0.87)
plt.show()
