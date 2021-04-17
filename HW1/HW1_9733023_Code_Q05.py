'''
___OVERVIEW___
CODE FOR QUESTION 5 - HW 1

___SUMMARY___
Rotation and affine tranformation practice

___AUTHOR___
Amirhossein Daraie — 9733023

___DATE___
26/03/2021
06/01/1400

'''

import numpy as np
import matplotlib.pyplot as plt
import cv2

def print_img_data(img,title='Image'):
    """This function prints some essential information about input image.  

    Args:
        img (numpy array image): Input image 
        title (str, optional): Title of image. Defaults to 'Image'.
    """
    print(f'________ {title} _______')
    if len(image.shape) == 2:
        print(f'min:{img.min()}  max:{img.max()}  shape:{img.shape}\
\nmean:{img.mean()}  std:{img.std()}  type:{type(img[0,0])}\n')
    elif len(image.shape) == 3:
        print(f'min:{img.min()}  max:{img.max()}  shape:{img.shape}\
\nmean:{img.mean()}  std:{img.std()}  type:{type(img[0,0,0])}\n')
        
    
def im_show(img_list,title=['']):
    """This function is an alias for showing an image with opencv in python

    Args:
        img_list (list of numpy array images): A sequence of images you want to display
        title (list, optional):A sequence of titles for your images. Defaults to [''].
    """
    Error = False
    if title == ['']:
        title=[f'Image {i+1}' for i in range(len(img_list))]
    if len(title) != len(img_list):
        print('ERROR 1 (im_show)')
        Error = True
    if not Error:
        for i,img in enumerate(img_list):
            cv2.imshow(title[i],img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

########## TASK 1 #######################################################
# load image as color image
image = cv2.imread("AUT-DIP.png", flags=cv2.IMREAD_GRAYSCALE)
# devide image to 6 sub-images
h, w = image.shape[0],image.shape[1]
image1 = image[0:h//2,0:w//3]
image2 = image[0:h//2,w//3:2*w//3]
image3 = image[0:h//2,2*w//3:w]
image4 = image[h//2:h,0:w//3]
image5 = image[h//2:h,w//3:2*w//3]
image6 = image[h//2:h,2*w//3:w]
# show 6 sub-images
# im_show([image1,image2,image3,image4,image5,image6])

########## TASK 2 #######################################################
# scale image1 with k=2
image1_t = cv2.resize(image1, (2*image1.shape[0], 2*image1.shape[1]))
# defines values for cropping the center of the image1
height, width = image1_t.shape[0], image1_t.shape[1]
new_height, new_width = image1.shape[0], image1.shape[1]
left = (width - new_width)//2
top = (height - new_height)//2
right = (width + new_width)//2
bottom = (height + new_height)//2
# crop the center of the image1
image1 = image1_t[top:bottom,left:right]
# show new image1
# im_show([image1,image1_t])

########## TASK 3 #######################################################
# Shear Horizontally
H, W = image2.shape
M2 = np.float32([[1, 0.2, 0], 
                 [0, 1, 0]])
# Use warpAffine to transform the image using the matrix, M2
image2 = cv2.warpAffine(image2, M2, (W, H))
# im_show([aff2])

########## TASK 4 #######################################################
H, W = image3.shape  
quarter_height, quarter_width = 100, -80
T = np.float32([[1, 0, quarter_width], 
                [0, 1, quarter_height]])
# Use warpAffine to transform the image using the matrix, T
image3 = cv2.warpAffine(image3, T, (W, H))
# im_show([img_translation])

########## TASK 5 #######################################################
H, W = image4.shape  
theta = 25 #degrees
T = np.float32([[np.cos(theta), -1*np.sin(theta), 0], 
                [np.sin(theta), np.cos(theta), 0]])
# Use warpAffine to transform the image using the matrix, T
image4 = cv2.warpAffine(image4, T, (W, H))
# im_show([img_rotation])

########## TASK 6 #######################################################
H, W = image5.shape  
theta = -25 #degrees
T = np.float32([[np.cos(theta), -1*np.sin(theta), 0], 
                [np.sin(theta), np.cos(theta), 0]])
# Use warpAffine to transform the image using the matrix, T
image5 = cv2.warpAffine(image5, T, (W, H))
# im_show([img_rotation2])

########## TASK 7 #######################################################
H, W = image6.shape  
theta = 45 #degrees
def rotateImage(image, angle):
    """Rotate an image 

    Args:
        image (numpy array (image)): Input image to rotate
        angle (float64): rotation degree

    Returns:
        numpy array (image): output image rotated by <angle> degress
    """
    center=tuple(np.array(image.shape[0:2])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    return rot_mat,cv2.warpAffine(image, rot_mat, image.shape[0:2],flags=cv2.INTER_LINEAR)
rot_mat, image6 = rotateImage(image6, theta)
print(rot_mat)
# im_show([img_rotation3])

########## TASK 8 #######################################################
final = np.zeros_like(image)
h, w = final.shape[0],final.shape[1]
final[0:h//2,0:w//3] = image1
final[0:h//2,w//3:2*w//3] = image2
final[0:h//2,2*w//3:w] = image3
final[h//2:h,0:w//3] = image4
final[h//2:h,w//3:2*w//3] = image5
final[h//2:h,2*w//3:w] = image6
# im_show([final])

########## TASK 9 #######################################################
# show final image
fig, ax = plt.subplots(1,figsize=(10,6))
st = fig.suptitle("Final Image", fontsize="x-large")
ax.imshow(final,cmap='gray', vmin=final.min(), vmax=final.max())
ax.axis("off")
plt.tight_layout()
plt.savefig('9733023-4.png', bbox_inches='tight')
plt.show()
