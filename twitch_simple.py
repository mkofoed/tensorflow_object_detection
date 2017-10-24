import cv2
import numpy as np
import time
import streamlink

# use live streamer to figure out the stream info
streams = streamlink.streams("http://www.twitch.tv/shroud")
stream = streams['best']
# open our out file. 
fname = "test.mpg"
vid_file = open(fname,"wb")
# dump from the stream into an mpg file -- get a buffer going
fd = stream.open()
for i in range(0,2*2048):
    if i%256==0:
        print("Buffering...")
    new_bytes = fd.read(1024)
    vid_file.write(new_bytes)
# open the video file from the begining
print("Done buffering.")
cam = cv2.VideoCapture(fname)
while True:
    _, img = cam.read()                      
    cv2.imshow('live_img',img)
    if (0xFF & cv2.waitKey(5) == 27) or img.size == 0:
        break
    # dump some more data to the stream so we don't run out. 
    new_bytes = fd.read(1024*16)
    vid_file.write(new_bytes)
vid_file.close()
fd.close()