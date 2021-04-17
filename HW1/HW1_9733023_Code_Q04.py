'''
___OVERVIEW___
CODE FOR QUESTION 4 - HW 1

___SUMMARY___
Image quantization and thresholding

___AUTHOR___
Amirhossein Daraie — 9733023

___DATE___
25/03/2021
05/01/1400

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
        
def quantize(image, bits=2):
    """This function digitalizes image with number of inputs bits.

    Args:
        image (numpy array (image)): Input image the you want to quantize
        bits (int, optional): Number of bits in final quantized image. Defaults to 2.

    Returns:
        numpy array (image): Quantized Image. 
        Example: for 2 bits, the values are 0,1,2,3
    """    
    div = 2**(8-bits)
    # note: you can imshow 'quantized_image' (0,1,2,...,255) not 'quantized' (0,1,2,3,...,10 e.g.)!
    quantized_image = image // div * div + div // 2
    # *** rank transform the quantized image ***
    quantized = np.copy(quantized_image)
    _,idx = np.unique(quantized,return_inverse=True)
    quantized = (idx.max() - idx + 1).reshape(quantized.shape)
    quantized = quantized.max()-1*quantized
    return quantized

# load image as color image
image = cv2.imread("Cube.tif", flags=cv2.IMREAD_COLOR)
# show color image
im_show([image])
# print color image data 
print_img_data(image)
# convert color image to grayscale image 
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# show grayscale image
im_show([image])
# print grayscale image data 
print_img_data(image)

# show image
fig, ax = plt.subplots(2,3,figsize=(10,6))
st = fig.suptitle("Image Quantization and Thresholding", fontsize="x-large")
qimage0 = quantize(image,bits=8)
ax[0,0].imshow(qimage0,cmap='gray', vmin=qimage0.min(), vmax=qimage0.max())
ax[0,0].set_title('8 Bits')
qimage1 = quantize(image,bits=5)
ax[0,1].imshow(qimage1,cmap='gray', vmin=qimage1.min(), vmax=qimage1.max())
ax[0,1].set_title('5 Bits')
qimage2 = quantize(image,bits=3)
ax[0,2].imshow(qimage2,cmap='gray', vmin=qimage2.min(), vmax=qimage2.max())
ax[0,2].set_title('3 Bits')
qimage3 = quantize(image,bits=2)
ax[1,0].imshow(qimage3,cmap='gray', vmin=qimage3.min(), vmax=qimage3.max())
ax[1,0].set_title('2 Bits')
qimage4 = quantize(image,bits=1)
ax[1,1].imshow(qimage4,cmap='gray', vmin=qimage4.min(), vmax=qimage4.max())
ax[1,1].set_title('1 Bit')
# threshold image
ret,thresh1 = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
ax[1,2].imshow(thresh1,cmap='gray', vmin=thresh1.min(), vmax=thresh1.max())
ax[1,2].set_title('Threshold')
for x in ax.ravel():
    x.axis("off")
plt.tight_layout()
plt.savefig('9733023-3.png', bbox_inches='tight')
plt.show()
