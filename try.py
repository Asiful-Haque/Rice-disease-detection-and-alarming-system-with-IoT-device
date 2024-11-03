import cv2

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam was opened successfully
if not cap.isOpened():
    print("Unable to open webcam")
    exit()

# Define a function to save a frame
def save_frame(frame, filename):
    cv2.imwrite(filename, frame)

# Start a loop to capture and save frames
while True:
    ret, frame = cap.read()

    # Check if a frame was read successfully
    if not ret:
        print("Unable to read frame from webcam")
        break

    # Save the frame
    #save_frame(frame, "frame.jpg")

    # Display the frame
    cv2.imshow("Webcam", frame)

    # Press `q` to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()

# Close all windows
cv2.destroyAllWindows()
