import cv2
import os


def main():
    vid=cv2.VideoCapture('/storage/emulated/0/raw_videos/')

    cur=0

    success=True
    path='/storage/emulated/0/Download'
    fps = round(vid.get(cv2.CAP_PROP_FPS))
    print(fps)

    while(True):
        success,frame=vid.read()
        if(success==False):
            break
        cv2.imwrite(os.path.join(path , 'frame%d.jpg'%cur), frame)
        
        
        cur+=1

    