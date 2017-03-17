import cv2
import numpy as np

def nothing(x):
    pass


image = np.zeros((300,512,3), np.uint8)
cv2.namedWindow('Window')


cv2.createTrackbar('R','Window',0,255,nothing)
cv2.createTrackbar('G','Window',0,255,nothing)
cv2.createTrackbar('B','Window',0,255,nothing)


switch = 'Toggle'
cv2.createTrackbar(switch, 'Window',0,1,nothing)

while(1):
    cv2.imshow('Window',image)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break


    r = cv2.getTrackbarPos('R','Window')
    g = cv2.getTrackbarPos('G','Window')
    b = cv2.getTrackbarPos('B','Window')
    s = cv2.getTrackbarPos(switch,'Window')

    if s == 0:
        image[:] = 0
    else:
        image[:] = [b,g,r]

cv2.destroyAllWindows()
