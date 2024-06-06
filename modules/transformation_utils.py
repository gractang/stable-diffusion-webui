import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import binary_erosion, binary_dilation
import json

def load_image(filename):
    img = cv2.imread(filename, 0)
    return img

# function to add two masks
def add_masks(mask1, mask2):
    return np.maximum(mask1, mask2)

def rescale_img(img, contours, scale_factor_x, scale_factor_y):
    """
    Rescales the object in both contour and mask images where the object is defined by black and white pixels.

    Parameters:
    - contour_image: Input binary contour image (numpy array).
    - mask_image: Input mask image (numpy array) with the same size as the contour image.
    - scale_factor_x: Factor by which to scale the object in the x direction.
    - scale_factor_y: Factor by which to scale the object in the y direction.

    Returns:
    - Rescaled contour image.
    - Rescaled mask image.
    """

    # Get the bounding box of the object
    x, y, w, h = cv2.boundingRect(contours[0])
    
    # # plot and save bounding box on image
    # cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)
    # cv2.imwrite("imgs/bounding_box.png", img)

    # Crop the object from the contour and mask images
    roi = img[y:y+h, x:x+w]

    # Rescale the object
    new_w = int(w * scale_factor_x)
    new_h = int(h * scale_factor_y)
    rescaled_img = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Create new images to place the rescaled objects
    new_image = np.zeros_like(img)
    
    # Compute the position to place the rescaled object in the center
    start_x = x + (w - new_w) // 2
    start_y = y + (h - new_h) // 2
    
    # Handle cases where the rescaled object exceeds the original image boundaries
    start_x = max(start_x, 0)
    start_y = max(start_y, 0)
    end_x = min(start_x + new_w, new_image.shape[1])
    end_y = min(start_y + new_h, new_image.shape[0])
    
    # Calculate the region of interest for placing
    roi_img = rescaled_img[:end_y-start_y, :end_x-start_x]
    
    # Place the rescaled object back in the new image
    new_image[start_y:end_y, start_x:end_x] = roi_img

    return new_image

def rescale_contour_and_mask(contour_image, mask_image, scale_factor_x, scale_factor_y):
    _, binary_image = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No contours found in the image")

    rescaled_contour = rescale_img(contour_image, contours, scale_factor_x, scale_factor_y)
    rescaled_mask = rescale_img(mask_image, contours, scale_factor_x, scale_factor_y)
    return rescaled_contour, rescaled_mask

def rotate_object_in_image(image, contours, angle):
    """
    Rotates an object within an image by a given angle.

    Parameters:
    - image: Input image (numpy array).
    - angle: Angle by which to rotate the object (in degrees).

    Returns:
    - Rotated image with the object.
    """
    # Get the bounding box of the object
    x, y, w, h = cv2.boundingRect(contours[0])

    # Crop the object from the image
    object_roi = image[y:y+h, x:x+w]

    # Compute the center of the object
    center = (w // 2, h // 2)
    
    # Compute the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Perform the rotation on the cropped object
    rotated_object = cv2.warpAffine(object_roi, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # Calculate the size of the new bounding box
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust the rotation matrix to take into account the new bounding box size
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # Perform the rotation on the cropped object with the adjusted matrix
    rotated_object = cv2.warpAffine(object_roi, M, (new_w, new_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # Create a new blank image with the same size as the original image
    new_image = np.zeros_like(image)
    
    # Calculate the position to place the rotated object in the new image
    start_x = max(0, x + (w - new_w) // 2)
    start_y = max(0, y + (h - new_h) // 2)
    end_x = min(start_x + new_w, new_image.shape[1])
    end_y = min(start_y + new_h, new_image.shape[0])
    
    # Calculate the region of interest in the rotated object
    roi_start_x = max(0, -x - (w - new_w) // 2)
    roi_start_y = max(0, -y - (h - new_h) // 2)
    roi_end_x = roi_start_x + (end_x - start_x)
    roi_end_y = roi_start_y + (end_y - start_y)
    
    # Place the rotated object back in the new image
    new_image[start_y:end_y, start_x:end_x] = rotated_object[roi_start_y:roi_end_y, roi_start_x:roi_end_x]

    return new_image

def rotate_contour_and_mask(contour_image, mask_image, angle):
    _, binary_image = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No contours found in the image")

    rotated_contour = rotate_object_in_image(contour_image, contours, angle)
    rotated_mask = rotate_object_in_image(mask_image, contours, angle)
    return rotated_contour, rotated_mask

def translate_object_in_image(image, contours, translate_x, translate_y):
    """
    Translates an object within an image by given x and y distances.

    Parameters:
    - image: Input image (numpy array).
    - translate_x: Distance by which to translate the object along the x-axis.
    - translate_y: Distance by which to translate the object along the y-axis.

    Returns:
    - Translated image with the object.
    """
    # Get the bounding box of the object
    x, y, w, h = cv2.boundingRect(contours[0])

    # Crop the object from the image
    object_roi = image[y:y+h, x:x+w]

    # Create a new blank image with the same size as the original image
    new_image = np.zeros_like(image)
    
    # Calculate the new position to place the translated object
    new_x = x + translate_x
    new_y = y + translate_y
    
    # Ensure the new position is within the image boundaries
    start_x = max(0, new_x)
    start_y = max(0, new_y)
    end_x = min(start_x + w, new_image.shape[1])
    end_y = min(start_y + h, new_image.shape[0])
    
    # Calculate the regions of interest for placing
    roi_object = object_roi[:end_y-start_y, :end_x-start_x]
    
    # Place the translated object back in the new image
    new_image[start_y:end_y, start_x:end_x] = roi_object

    return new_image

def translate_contour_and_mask(contour_image, mask_image, translate_x, translate_y):
    _, binary_image = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No contours found in the image")

    translated_contour = translate_object_in_image(contour_image, contours, translate_x, translate_y)
    translated_mask = translate_object_in_image(mask_image, contours, translate_x, translate_y)
    return translated_contour, translated_mask

def transform_contour_and_mask(contour_image, mask_image, scale_factor_x, scale_factor_y, angle, translate_x, translate_y):
    rescaled_contour, rescaled_mask = rescale_contour_and_mask(contour_image, mask_image, scale_factor_x, scale_factor_y)
    rotated_contour, rotated_mask = rotate_contour_and_mask(rescaled_contour, rescaled_mask, angle)
    translated_contour, translated_mask = translate_contour_and_mask(rotated_contour, rotated_mask, translate_x, translate_y)
    # translated_contour, translated_mask = rotated_contour, rotated_mask # for debugging
    return translated_contour, translated_mask

def shrink_mask(mask, n):
    """
    Shrinks the True areas in a boolean mask by 'n' pixels.

    Parameters:
        mask (np.ndarray): A 2D NumPy array of boolean values, where True indicates the masked area.
        n (int): The number of pixels by which to shrink the mask.

    Returns:
        np.ndarray: A new mask array with the True areas shrunk by 'n' pixels.
    """
    # Create a structuring element that defines the "connectivity" or shape of the erosion
    # For example, a square structuring element with size (2*n+1, 2*n+1)
    structuring_element = np.ones((2*n+1, 2*n+1), dtype=bool)
    
    # Apply binary erosion
    shrunk_mask = binary_erosion(mask, structure=structuring_element)
    
    return shrunk_mask

def create_new_contour(transformed_contour, transformed_mask, original_contour, original_mask, n=2):
    shrunk_rescaled_mask = shrink_mask(transformed_mask, n)
    shrunk_orig_mask = shrink_mask(original_mask, n)
    new_contour_img = (1 - shrunk_orig_mask) * original_contour + shrunk_rescaled_mask * transformed_contour
    return new_contour_img

def read_params(filepath, debug=False):
    """
    Reads a json file with the relevant params and returns
    scale_factor_x, scale_factor_y, angle, translate_x, translate_y
    """
    if debug:
        return 1, 1, 0, 0, 0
    params = json.load(open(filepath))
    scale_factor_x = params["scale_factor_x"]
    scale_factor_y = params["scale_factor_y"]
    angle = params["angle"]
    translate_x = params["translate_x"]
    translate_y = params["translate_y"]
    return scale_factor_x, scale_factor_y, angle, translate_x, translate_y

def get_final_contour_and_mask(contour_image, mask_image, filepath):
    """
    Gets the final contour and mask after applying the transformation
    """
    scale_factor_x, scale_factor_y, angle, translate_x, translate_y = read_params(filepath)
    print("Params:", scale_factor_x, scale_factor_y, angle, translate_x, translate_y)
    transformed_contour, transformed_mask = transform_contour_and_mask(contour_image, mask_image, scale_factor_x, scale_factor_y, angle, translate_x, translate_y)
    final_contour = create_new_contour(transformed_contour, transformed_mask, contour_image, mask_image)
    return final_contour, add_masks(mask_image, transformed_mask)