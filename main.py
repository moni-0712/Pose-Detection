import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Initialize the video capture
cap = cv2.VideoCapture('3.mp4')

# Set the desired size for the resized video frame
desired_width = 640
desired_height = 480

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break
    
    # Convert the BGR image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect poses
    results = pose.process(img_rgb)
    
    # Draw pose landmarks on the frame
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # Resize the frame to the desired size
    img_resized = cv2.resize(img, (desired_width, desired_height))
    
    # Create a blank canvas
    canvas_height, canvas_width = 1080, 1920  # For example, Full HD canvas
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    
    # Calculate top-left corner to place the resized frame in the center of the canvas
    x_offset = (canvas_width - desired_width) // 2
    y_offset = (canvas_height - desired_height) // 2
    
    # Place the resized frame in the center of the canvas
    canvas[y_offset:y_offset + desired_height, x_offset:x_offset + desired_width] = img_resized
    
    # Display the canvas with the resized frame in the center
    cv2.imshow('posedetection', canvas)
    
    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
