import cv2
import mask_validator

img = cv2.imread('test_image.jpg')
# use cropped face image
print(mask_validator.validate_mask(img))
