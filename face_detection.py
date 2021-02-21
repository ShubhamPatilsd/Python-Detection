import cv2
import sys


#get the resources given to us on execution (python3 face_detection.py abba.png haarcascade_frontalface_default.xml)
imagePath = sys.argv[1]
cascPath = sys.argv[2]

#create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

#Read the image and convert it to grayscale
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#detect faces in the image
faces = faceCascade.detectMultiScale(
    #pass in the grayscale image
    gray,
    #Since some faces may be closer to the camera, they would appear bigger than the faces in the back. The scale factor compensates for this.
    scaleFactor=1.3,
    # The detection algorithm uses a moving window to detect objects. minNeighbors defines how many objects are detected near the current one before it declares the face found.
    minNeighbors=5,
    # Gives the size of each window
    minSize = (30,30),
    #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

#draw a rectangle around the faces
for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

# display image 
cv2.imshow("Faces found",image)
#wait for key press
cv2.waitKey(0)

