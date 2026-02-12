import numpy as np
import cv2

def preprocess_frame(frame):
    # Define the desired width and height of the frames after resizing
    width = 224
    height = 224

    # Define the normalization parameters
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Resize the frame to the desired dimensions
    resized_frame = cv2.resize(frame, (width, height))

    # Convert the frame to float32 data type
    normalized_frame = resized_frame.astype(np.float32)

    # Normalize the frame using the defined mean and standard deviation
    normalized_frame /= 255.0
    normalized_frame -= mean
    normalized_frame /= std

    return normalized_frame

#--------------------------------------------------------------------------------------------------------------------------------------#

def replaceNaN(frames_array_labeled, angles_array_labeled):
    # Check if np array contain NaN values
    # has_nan = np.isnan(angles_array_labeled).any()
    # if has_nan:
    #     print("The array contains NaN values.")
    # else:
    #     print("The array does not contain NaN values.")

    # Replace NaN values by 0
    angles_array_labeled = np.nan_to_num(angles_array_labeled, nan=0)

    has_nan = np.isnan(angles_array_labeled).any()
    if has_nan:
        print("The array contains NaN values.")
    else:
        print("The array does not contain NaN values.")
    
    # print type arrays
    print("Angles array type",angles_array_labeled.dtype)
    print("Frames array type",frames_array_labeled.dtype)

    return frames_array_labeled, angles_array_labeled