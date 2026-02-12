import numpy as np
import cv2
import preprocess as p


def getAngles():
    # Define the path to the labeled videos and corresponding label files
    videos_path = 'C:/Users/youss/Desktop/Youssef/Stage2/Tests/calib_challenge-main/labeled/'
    labels_path = 'C:/Users/youss/Desktop/Youssef/Stage2/Tests/calib_challenge-main/labeled/'

    # Define the dimensions of the frames (assumed to be the same for all videos)
    frame_width = 1920
    frame_height = 1080

    # Define the number of frames in each video (assumed to be the same for all videos)
    num_frames = 1200

    # Define the focal length of the camera
    focal_length = 910

    # nparay for all angles
    angles_labeled = []

    # frames
    frames = []

    # Initialize an array to store the pitch and yaw angles for each frame
    # angles = np.zeros((num_frames * 5, 2))
    angles = []

    # Load the labeled videos and extract the pitch and yaw angles for each frame
    for i in range(4):

        # Load the label file
        labels = np.loadtxt(labels_path + str(i) + '.txt')

        lenght_label = len(labels)
        diff_lenght = num_frames - lenght_label

        # Iterate over each frame in the video
        for j in range(lenght_label):

            # Estimate the pitch and yaw angles for the current frame using the labels
            pitch = labels[j, 0]
            yaw = labels[j, 1]

            angles.append((pitch,yaw))
        
        if diff_lenght > 0:
            for k in range(diff_lenght):
                angles.append((None,None))
            

        # Print the shape of the angles array for verification
        #print(f"Angles shape for video {i}: {angles.shape}")

    # Convert the list of frames to a numpy array
    angles_array_labeled = np.array(angles, dtype = np.float32)
    print(f"Angles shape for video : {angles_array_labeled.shape}")

    return angles_array_labeled

#--------------------------------------------------------------------------------------------------------------------------------------#

def getFrames():
    # Define the path to the labeled videos
    videos_path = 'C:/Users/youss/Desktop/Youssef/Stage2/Tests/calib_challenge-main/labeled/'

    # Define the dimensions of the frames (assumed to be the same for all videos)
    frame_width = 1920
    frame_height = 1080

    # Define the number of frames in each video (assumed to be the same for all videos)
    num_frames = 1200

    # Define a default frame to use for filling in missing frames
    default_frame = np.zeros((frame_height, frame_width, 3), dtype=np.float32)

    # Initialize a list to store the frames for each video
    frames_list = []

    # Iterate over each video
    for i in range(4):
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
                # If the frame was not read, use the default frame instead
                frame = default_frame

            frame = p.preprocess_frame(frame)
            
            # Append the frame to the list of frames for the current video
            #frames.append(frame)
            frames_list.append(frame)

        # Release the video object
        video.release()


        # Convert the list of frames to a numpy array
        frames_array_labeled = np.array(frames_list)
        #print(f"Frames shape for video {i}: {frames_array_labeled.shape}")

    # Convert the list of frames to a numpy array
    frames_array_labeled = np.array(frames_list)
    print(f"Frames shape for video : {frames_array_labeled.shape}")

    # Print the shape of the frames list for verification
    print(f"Frames shape: {len(frames_list)} videos, {len(frames_list[0])} frames per video, {len(frames_list[0])}x{len(frames_list[0][0])} pixels")

    return frames_array_labeled