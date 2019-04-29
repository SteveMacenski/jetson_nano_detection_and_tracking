import numpy as np
import cv2
import time
import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst


""" MIPI Camera Interface using gstreamer """
class MipiCamera():
    def __init__(self, width=300, height=300):
        self.cap = None
        self.width = width
        self.height = height

    def _gstStr(self, width, height):
        return 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=%d, height=%d, format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! appsink' % (3280, 2464, 21, width, height)

    def startStreaming(self):
        print ("Starting to stream camera...")
        self.cap = cv2.VideoCapture(self._gstStr(self.width, self.height))

    def getFrame(self):
        rtn_val, frame = self.cap.read()
        if rtn_val:
            return frame
        else:
            print ("Failed to capture frame!")
            return None

    def isOpened(self):
        if self.cap:
            return self.cap.isOpened()
        else:
            return False

    def __del__(self):
        if self.cap:
            self.cap.release()
        print ("Cleanly exited MipiCamera")


