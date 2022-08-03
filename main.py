import cv2
import mediapipe as mp
from itertools import cycle
import face_mesh
import pose
import segmentation
import three_d_object
import yolov4

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

display_options = ["yolo", "facemesh", "threed", "pose", "segmentation"]
display_cycle = cycle(display_options)
current_display = next(display_cycle)

# For webcam input:
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

    if current_display == "facemesh":
        image = face_mesh.process_facemesh(image)
        image = cv2.flip(image, 1)
    elif current_display == "threed":
        image = three_d_object.process_threed(image)
        image = cv2.flip(image, 1)
    elif current_display == "pose":
        image = pose.process_pose(image)
        image = cv2.flip(image, 1)
    elif current_display == "yolo":
        image = yolov4.process_yolo(image)
    elif current_display == "segmentation":
        image = cv2.flip(image, 1)
        image = segmentation.process_segmentation(image)

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('Insight Mastery', image)

    pressedKey = cv2.waitKeyEx(60)
    if pressedKey == -1:
        continue
    elif pressedKey == 63235:
        current_display = next(display_cycle)
    elif pressedKey == 121:
        current_display = "yolo"
    elif pressedKey == 115:
        current_display = "segmentation"
    elif pressedKey == 112:
        current_display = "pose"
    elif pressedKey == 102:
        current_display = "facemesh"
    elif pressedKey == 51:
        current_display = "threed"
    elif pressedKey == 27:
        break

cap.release()
