import numpy as np
import cv2
import preprocess as p
import cnn

def prediction(model):
    # Define the path to the unlabeled videos
    videos_path = 'C:/Users/youss/Desktop/Youssef/Stage2/Tests/calib_challenge-main/labeled/'

    # Define the path to the output files
    output_path = 'C:/Users/youss/Desktop/Youssef/Stage2/Tests/calib_challenge-main/folder/'

    # Define the dimensions of the frames (assumed to be the same for all videos)
    frame_width = 1920
    frame_height = 1080

    # Define the number of frames in each video (assumed to be the same for all videos)
    num_frames = 1200

    # file
    i = 4

    # Initialize arrays to store the predicted pitch and yaw angles for each frame in each video
    pitch_angles = np.zeros((5, num_frames))
    yaw_angles = np.zeros((5, num_frames))

    # Load the video
    video = cv2.VideoCapture(videos_path + str(i) + '.hevc')

    # Initialize an empty list to store the frames for the current video
    frames = []

    # Iterate over each frame in the video
    for j in range(num_frames):
        # Read the frame
        ret, frame = video.read()

        # Check if the frame was successfully read
        if not ret:
            break

        # Preprocess the frame
        frame = p.preprocess_frame(frame)

        # Make a prediction using the model
        prediction = model.predict(np.expand_dims(frame, axis=0))

        # Extract the pitch and yaw angles from the prediction
        pitch_angle, yaw_angle = prediction[0]

        # Store the pitch and yaw angles in the arrays
        pitch_angles[i, j] = pitch_angle
        yaw_angles[i, j] = yaw_angle

    # Release the video object
    video.release()

    # Write the pitch and yaw angles to files
    np.savetxt(output_path + str(i) + '.txt', np.vstack((pitch_angles[i], yaw_angles[i])).T)

