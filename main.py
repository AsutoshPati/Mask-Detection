import cv2
import mask_validator
import time

img = cv2.imread('test image path')
print(mask_validator.validate_mask(img))
t1 = time.time()
print(mask_validator.validate_mask(img))
print(time.time()-t1)
