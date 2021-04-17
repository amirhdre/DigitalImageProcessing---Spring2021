'''
___OVERVIEW___
CODE FOR QUESTION 3 - HW 1

___SUMMARY___
Operations to reduce noise 
Video operations

___AUTHOR___
Amirhossein Daraie — 9733023

___DATE___
25/03/2021
05/01/1400

'''

import numpy as np
import cv2

def print_img_data(img,title='Image'):
    print(f'________ {title} _______')
    print(f'min:{img.min()}  max:{img.max()}  shape:{img.shape}\
\nmean:{img.mean()}  std:{img.std()}  type:{type(img[0,0])}\n')
    
def im_show(img_list,title=['']):
    Error = False
    if title == '':
        title=[f'Image {i+1}' for i in range(len(img_list))]
    if len(title) != len(img_list):
        print('ERROR 1 (im_show)')
        Error = True
    if not Error:
        for i,img in enumerate(img_list):
            cv2.imshow(title[i],img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    

##########################################################################
## Part 1
## Subtract first frame of video from image to find noise
##########################################################################

# load image
image = cv2.imread("MRI-Head.png", flags=cv2.IMREAD_GRAYSCALE)
# set video file path of input video with name and extension
vid = cv2.VideoCapture('./MRI.avi')
# Extract first frame
ret, frame1 = vid.read()
# convert image to Grayscale
frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
print_img_data(frame1,'Frame1')
print_img_data(image,'Image')
# subtract frame from image
noise = np.add(frame1,-1*image)
# show noise
print_img_data(noise,'Noise')
# find mean and std of noise
noise_mean = noise.mean()
noise_std = noise.std()
# deallocates memory and clears capture pointer
vid.release()
# show image
im_show([frame1,image],
        ['Frame 1','Image'])
cv2.waitKey(0)
cv2.destroyAllWindows()

##########################################################################
## Part 2
## Calculate mean of all frames in video
##########################################################################

# set video file path of input video with name and extension
cap = cv2.VideoCapture('./MRI.avi')
image_averaged = np.zeros((256,256))
# for frame identity
index = 0
while(True):
    # Extract images
    ret, frame_i = cap.read()
    # end of frames
    if not ret: 
        break
    # convert image to Grayscale
    frame_i = cv2.cvtColor(frame_i, cv2.COLOR_BGR2GRAY)
    # Saves images and convert image to int32
    image_averaged = (np.add(image_averaged,frame_i)).astype(np.int32)
    index += 1
# deallocates memory and clears capture pointer
cap.release()
# take mean of all 500 frames 
image_averaged = image_averaged / index
# show image
# If the type is float, the values must be between 0 and 1 to be displayed correctly
# So scale it to be between 0 and 1
im_show([image_averaged/255],['image_averaged'])

#####################################
## Part 3
## Subtract averaged image from first frame of video to find noise
#####################################

print_img_data(frame1,'Frame1')
print_img_data(image_averaged,'image_averaged')

noise2 = np.add(image_averaged,-1*image)
# find mean and std of noise
noise2_mean = noise2.mean()
noise2_std = noise2.std()
print_img_data(noise2,'noise2')


##########################################################################
## Part 4
## Mask image
##########################################################################

# load mask
mask = cv2.imread("mask.png", flags=cv2.IMREAD_GRAYSCALE)
# mask image
image_masked = cv2.bitwise_and(image,image,mask = mask)
# show image
im_show([image_masked],['image_masked'])