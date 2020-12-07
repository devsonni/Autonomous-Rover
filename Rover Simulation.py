import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
image = mpimg.imread('..\example_grid1.jpg')
plt.imshow(image)
plt.show() 

def perspect_transform(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    return warped
dst_size = 5 
bottom_offset = 6
source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
                  [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  ])

warped = perspect_transform(image, source, destination)

cv2.polylines(image, np.int32([source]), True, (0, 0, 255), 3)
cv2.polylines(warped, np.int32([destination]), True, (0, 0, 255), 3)
               
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 6), sharey=True)
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=40)

ax2.imshow(warped, cmap='gray')
ax2.set_title('Result', fontsize=40)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

image_name = '..\sample.jpg'
image = mpimg.imread(image_name)
plt.imshow(image)
plt.show()
#print(image.dtype,image.shape,np.min(image),np.max(image))
red_channel = np.copy(image)
green_channel = np.copy(image)
blue_channel = np.copy(image)
red_channel[:,:,[1,2]] = 0
blue_channel[:,:,[0,1]] = 0
green_channel[:,:,[0,2]] = 0
fig = plt.figure(figsize=(12,3))
plt.subplot(131)
plt.imshow(red_channel)
plt.subplot(132)
plt.imshow(green_channel)
plt.subplot(133)
plt.imshow(blue_channel)
plt.show()
def color_thresh(img, rgb_thresh = (0, 0, 0)):
    above_thresh = (img[:,:,0] > rgb_thresh[0])  \
                    & (img[:,:,1] > rgb_thresh[1]) \
                    & (img[:,:,2] > rgb_thresh[2])
    color_select = np.zeros_like(img[:,:,0])
    color_select[above_thresh] = 1
    return color_select
    
red_threshold = 160
green_threshold = 160
blue_threshold = 160

rgb_threshold = (red_threshold, green_threshold, blue_threshold)

# pixels below the thresholds
colorsel = color_thresh(image, rgb_thresh = rgb_threshold)

# Display the original image and binary               
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(21, 7), sharey=True)
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=40)

ax2.imshow(colorsel, cmap='gray')
ax2.set_title('Your Result', fontsize=40)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
