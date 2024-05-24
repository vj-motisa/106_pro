import cv2


# Create our body classifier
body_classifier = cv2.CascadeClassifier("C:/Users/Admin/OneDrive/Desktop/Python/106_pro/haarcascade_fullbody.xml")

# Initiate video capture for video file
cap = cv2.VideoCapture('C:/Users/Admin/OneDrive/Desktop/Python/106_pro/walking.avi')

# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = cap.read()

    #Convert Each Frame into Grayscale
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # Pass frame to our body classifier
    bodies = body_classifier.detectMultiScale(gray,1.2,3)
    
    # Extract bounding boxes for any bodies identified
    for x,y,w,h in bodies:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        crop = frame[y:y+h,x:x+w]
        cv2.imwrite("faces.jpg",crop)
    cv2.imshow("Project 106", frame)

    if cv2.waitKey(1) == 32: #32 is the Space Key
        break

cap.release()
cv2.destroyAllWindows()
