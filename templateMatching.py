__author__ = 'GAK'
import cv2
import numpy as np
import time
import sys

def diffImg(t0, t1, t2):
  d1 = cv2.absdiff(t2, t1)
  d2 = cv2.absdiff(t1, t0)
  return cv2.bitwise_and(d1, d2)


def filterMatches( kp1, kp2, matches, ratio = 0.75 ):
    mkp1, mkp2 = [], []
    for m in matches:
        if len( m ) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )

    pairs = zip( mkp1, mkp2 )

    return pairs

template = cv2.imread('1.jpg',0)

edges = cv2.Canny(template, 50, 200)
(tH, tW) = edges.shape[:2]
cv2.imshow("Template", edges)





cam = cv2.VideoCapture(1)

winName = "Movement Indicator"
cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)


#cascPath = sys.argv[0]
#faceCascade = cv2.CascadeClassifier(cascPath)

# Read three images first:
#t_minus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
t_minus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
time.sleep(.25)
t = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
time.sleep(.25)
t_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
time.sleep(.25)
t0, filenum = 0, 1

while True:
    time.sleep(.25)
    cv2.imshow( winName, diffImg(t_minus, t, t_plus) )

  # Read next image

    t_minus = t
    t = t_plus
    t_plus_new = cv2.cvtColor(cam.read()[1], cv2.COLOR_BGR2GRAY)
    isTure=np.equal(np.array(t_plus_new).all(),  np.array(t_plus).all())
    t_plus=t_plus_new

    #if(diffImg(t_minus, t, t_plus).any()):
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('frame',gray)

    if(isTure):

        print("Motion Detected matching with Object")

        #cv2.imwrite(str(filenum) + ".jpg", frame)
        #cv2.imwrite("image/"+str(filenum) + ".jpg", frame)
        filenum +=1
        orb = cv2.ORB_create()


        kp1, des1 = orb.detectAndCompute(template,None)
        kp2, des2 = orb.detectAndCompute(frame,None)


        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1,des2)



        matches = sorted(matches, key = lambda x:x.distance)
        outImg=template

        sorted(matches, reverse=True)


        if(len(matches) > 500):
             print( "matches has", len(matches))
             img3 = cv2.drawMatches(template,kp1,frame,kp2,matches[:10], outImg, flags=2)
             cv2.imwrite("image/tes"+str(filenum) + ".jpg", img3)
             break
               # FLANN parameters
        else:
            print("no definate mateches", len(matches))
            time.sleep(.10)


    else:
       t_plus=t_plus_new
       print("Not eaual")



       #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       #cv2.imshow('frame',gray)


    key = cv2.waitKey(10)

    if key == 27:
       cv2.destroyWindow(winName)
       break

print "Goodbye"