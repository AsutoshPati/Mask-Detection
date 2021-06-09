# Mask Detection


The model has been trained with [google teachable machine](https://teachablemachine.withgoogle.com/train/image) with 
data collected by [Asutosh Pati](https://in.linkedin.com/in/asutoshpati).
Just put the mask_validator.py in your project directory to use it and import the file in your main program to use it. 
You can use the example file and image to test the program. 

## Requirements
numpy==1.19.5  
opencv-contrib-python==4.5.1.48  
opencv-python==4.5.1.48  
tensorflow-gpu==2.5.0  

## How to use
```python
import cv2
import mask_validator

img = cv2.imread('test_image.jpg')
# use cropped face image
print(mask_validator.validate_mask(img))
```

#### validate_mask function
This function predicts the whether mask is present in cropped face image or not. It
returns the prediction label(Without Mask/With Mask) & confidence (in 0 to 1) of prediction.

**Parameter**
* image: numpy.ndarray  
  &emsp; Cropped face image as numpy array.  

**Returns**
* op_class: string  
  &emsp; Output class label of predicted image.
* op_factor: float  
  &emsp; Output confidence (percentage) of prediction ranges in 0 to 1.
  