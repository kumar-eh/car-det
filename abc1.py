import cv2 as cv
from datetime import datetime
import os
vid = cv.VideoCapture("vidoes/test4.mp4")
currentframe =0 
if not os.path.exists('frames_videos'):
    os.makedirs('frames_videos')
# this is suitable for stable cameras we can apply parameters too on the detector algorihtms
# object_detector = cv.createBackgroundSubtractorMOG2()
# more stable the camera more history we can give and the varithreshold is flase positives
object_detector = cv.createBackgroundSubtractorMOG2(history=300 , varThreshold=300)
# changing directory to given path

while True:
    def bounding_box_img(img,bbox):
        x_min, y_min, x_max, y_max = bbox
        bbox_obj = img[y_min:y_max, x_min:x_max]
        return bbox_obj
    isTrue , frame = vid.read()
    # height , width , _ = frame.shape
    # print(height, width)
    roi = frame[300:720 , 100:500]
    roi1 = frame[500 : 720 , 0:500]
    mask = object_detector.apply(roi)
    # remove everythong that is not white
    _ , mask = cv.threshold(mask , 254, 255,cv.THRESH_BINARY)
    contours , _ =cv.findContours(mask , cv.RETR_EXTERNAL , cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if(area>100):
            # cv.drawContours(roi, [cnt]  , -1 , (0,0,255) , 2)
            # drawing rectangle on the detectedd objects
            x,y,w,h = cv.boundingRect(cnt)
            cv.rectangle(roi , (x,y), (x+w , y+h) , (0,0,255 ), 2 )
            cropped_image = roi1[y:y+h, x:x+w]
            area = w*h
            # save only the pics where the area is above certain limit which is size of a car approx
            if(area>18000):
                cv.imwrite('.frames_videos' + str(currentframe) + '.jpg' ,roi1 )
                currentframe+=1
    # cv.imshow('roi' , roi)
    cv.imshow('vidoe save' , roi1)
    cv.imshow('Video' , frame)
    # cv.imshow('MAsk' , mask)
    if(cv.waitKey(20) & 0xFF == ord('d')):
        break
vid.release()
cv.destroyAllWindows()