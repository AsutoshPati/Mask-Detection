"""
    Project Name        : Mask Detection
    Author              : Asutosh Pati
    Date of Creation    : 14 MAY 2021
    Purpose             : Detect whether face image has mask or not.
    Description         :
    Version             : ver 1.0.0
    Modifications       :
        MOD-0000-DT-yyyy_mm_dd : Modification done by, Modification ticket No.
                                 Description
"""

from tensorflow.keras import models
import numpy as np
import cv2
import gc

model = models.load_model('model/keras_model.h5')   # load the mask detection model
labels = ['No Mask', 'Mask']                        # label classes during training


def preprocess_image(image: np.ndarray):
    """
    This function takes the cropped face image as input and preprocess the image which will
    be converted to normalized data, which can be fitted in prediction model.

    Parameters
    ----------
    image: np.ndarray
        Cropped face image.

    Returns
    -------
    data: np.ndarray
        The normalized image data to be applied in prediction model.

    """
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image_array = cv2.resize(image, (224, 224))
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    return data


def validate_mask(image: np.ndarray):
    """
    This function predicts the whether mask is present in cropped face image or not. It
    returns the prediction label(Mask/No Mask) & confidence (in 0 to 1) of prediction.

    Parameters
    ----------
    image: np.ndarray
        Cropped face image.

    Returns
    -------
    op_class: str
        Output class label of predicted image.
    op_factor: float
        Output confidence (percentage) of prediction ranges in 0 to 1.

    """
    data = preprocess_image(image)
    prediction = list(model.predict(data)[0])
    op_factor = max(prediction)
    op_class = labels[prediction.index(op_factor)]
    gc.collect()
    return op_class, op_factor

