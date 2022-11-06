import cv2
import os
def hello(something):
    vid=cv2.VideoCapture('/storage/emulated/0/Movies/CameraX-Video/20_abc_.mp4')
    cur=0
    success=True
    path='/storage/emulated/0/Download'
    fps = vid.get(cv2.CAP_PROP_FRAME_COUNT)

    while(True):
        success,frame=vid.read()
        if(success==False):
            return cur
            break
        cv2.imwrite(os.path.join(path , 'frame%d.jpg'%cur), frame)
        
        
        cur+=1
    



