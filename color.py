import cv2
import numpy as np

def changeHueMin(value):
    global hueMin
    hueMin = value

def changeHueMax(value):
    global hueMax
    hueMax = value

def changeSatMin(value):
    global satMin
    satMin = value

def changeSatMax(value):
    global satMax
    satMax = value

def changeValMin(value):
    global valMin
    valMin = value

def changeValMax(value):
    global valMax
    valMax = value

hueMin = 0
satMin = 0
valMin = 0
hueMax = 255
satMax = 255
valMax = 255

window_name = "TrackBars"
cv2.namedWindow(window_name)

cv2.createTrackbar("HueMinTrackbar", window_name, hueMin, 255, changeHueMin)
cv2.createTrackbar("HueMaxTrackbar", window_name, hueMax, 255, changeHueMax)
cv2.createTrackbar("SatMinTrackbar", window_name, satMin, 255, changeSatMin)
cv2.createTrackbar("SatMaxTrackbar", window_name, satMax, 255, changeSatMax)
cv2.createTrackbar("ValMinTrackbar", window_name, valMin, 255, changeValMin)
cv2.createTrackbar("ValMaxTrackbar", window_name, valMax, 255, changeValMax)

capture = cv2.VideoCapture(1)

while True:
    frameAvailable, frame = capture.read()
    if not frameAvailable:
        break
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([hueMin, satMin, valMin])
    higher = np.array([hueMax, satMax, valMax])
    mask = cv2.inRange(hsv, lower, higher)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow(window_name, frame)
    cv2.imshow("Maske", mask)
    cv2.imshow("Result", result)

    key = cv2.waitKey(10)
    if key >= 0:
        break

cv2.destroyAllWindows()