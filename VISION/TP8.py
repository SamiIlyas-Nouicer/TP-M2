import cv2 
import numpy as np
import time

cap = cv2.VideoCapture(2)
cap1 = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("output.avi",fourcc,60,(frame_width,frame_height))

if not cap.isOpened():
    print("error capture")
    exit(0)
    
    
while(cap.isOpened()):
    ret ,frame = cap.read()
    ret1 ,frame1 = cap1.read()
    t_start = time.time()
    if not ret:
        print("error read frame")
        break
    
    cv2.flip(frame,1,frame)

    out.write(frame)
    cv2.imshow("image",frame)
    cv2.imshow("image1",frame1)
    if cv2.waitKey(20) & 0xFF ==ord("q"):
        break
    print("FPS: ",1/(time.time()-t_start))
    # print("Time: ",time.time()-t_start) 
out.release()
cap.release()
cv2.destroyAllWindows()