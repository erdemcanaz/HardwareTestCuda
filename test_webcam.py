import cv2

# Create a VideoCapture object to access the webcam
cap = cv2.VideoCapture(0)


desired_width = 640
desired_height = 480
cap.set(3, desired_width)  # Set the width of the frame
cap.set(4, desired_height)  # Set the height of the frame

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    print(frame.shape)


    # Display the frame
    cv2.imshow('Webcam', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close the window
cap.release()
cv2.destroyAllWindows()