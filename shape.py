import cv2

# Open the RTSP stream
cap = cv2.VideoCapture("rtsp://admin:anas1155@46.152.196.211:554/Streaming/Channels/1?tcp/")

# Read the first frame to get its shape
ret, frame = cap.read()
if ret:
    print("Video shape:", frame.shape)
else:
    print("Failed to read frame")

# Release the capture
cap.release()

