import cv2
from mtcnn import MTCNN

# Initialize video capture and MTCNN detector
cap = cv2.VideoCapture(0)
detector = MTCNN()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Detect faces in the frame
    output = detector.detect_faces(frame)

    for single_output in output:
        # Extract bounding box coordinates
        x, y, width, height = single_output['box']
        
        # Draw rectangle around the face
        cv2.rectangle(frame, pt1=(x, y), pt2=(x + width, y + height), color=(255, 0, 0), thickness=2)

        # Extract keypoints
        keypoints = single_output['keypoints']
        left_eyeX, left_eyeY = keypoints['left_eye']
        right_eyeX, right_eyeY = keypoints['right_eye']
        noseX, noseY = keypoints['nose']
        mouth_leftX, mouth_leftY = keypoints['mouth_left']
        mouth_rightX, mouth_rightY = keypoints['mouth_right']

        # Draw circles on keypoints
        cv2.circle(frame, center=(left_eyeX, left_eyeY), color=(0, 255, 0), thickness=3, radius=2)
        cv2.circle(frame, center=(right_eyeX, right_eyeY), color=(0, 255, 0), thickness=3, radius=2)
        cv2.circle(frame, center=(noseX, noseY), color=(0, 255, 0), thickness=3, radius=2)
        cv2.circle(frame, center=(mouth_leftX, mouth_leftY), color=(0, 255, 0), thickness=3, radius=2)
        cv2.circle(frame, center=(mouth_rightX, mouth_rightY), color=(0, 255, 0), thickness=3, radius=2)

    # Display the resulting frame
    cv2.imshow('Face Detection with Keypoints', frame)

    # Break loop on 'x' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
