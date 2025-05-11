import cv2
# Load the Haar Cascade for face detection
face_cap = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open the default camera (0)
video_cap = cv2.VideoCapture(0)

while True:
    ret, video_data = video_cap.read()
    video_data = cv2.flip(video_data, 1)
    # Convert the frame to grayscale
    col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cap.detectMultiScale(
        col,
        scaleFactor= 1.1,
        minNeighbors= 5,
        minSize=(30, 30),
        flags= cv2.CASCADE_SCALE_IMAGE
    )

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    if not ret:
        print("Failed to grab frame")
        break

    # Show the video with detected faces
    
    cv2.imshow("Video Live", video_data)

    # Break the loop if 's' is pressed
    if cv2.waitKey(10) == ord("q"):
        break

# Release the video capture object and close the display window
video_cap.release()
cv2.destroyAllWindows()
