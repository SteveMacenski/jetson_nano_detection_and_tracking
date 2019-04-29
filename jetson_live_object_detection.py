import tensorflow as tf
import numpy as np
import cv2
import tensorflow.contrib.tensorrt as trt
import time
import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
from src.mipi_camera import MipiCamera
from src.object_detector import ObjectDetection


""" Jetson Live Object Detector """
class JetsonLiveObjectDetection():
    def __init__(self, model,  debug=False):
        self.debug = debug
        self.camera = MipiCamera(300, 300)
        self.model = model
        self.detector = ObjectDetection('../data/' + self.model)

    def _visualizeDetections(self, img, scores, boxes, classes, num_detections):
        cols = img.shape[1]
        rows = img.shape[0]
        detections = []

        for i in range(num_detections):
            bbox = [float(p) for p in boxes[i]]
            score = float(scores[i])
            classId = int(classes[i])
            if score > 0.5:
                x = int(bbox[1] * cols)
                y = int(bbox[0] * rows)
                right = int(bbox[3] * cols)
                bottom = int(bbox[2] * rows)
                thickness = int(4 * score)
                cv2.rectangle(img, (x, y), (right, bottom), (125,255, 21), thickness=thickness)
                detections.append(self.detector.labels[str(classId)])

        print ("Debug: Found objects: " + str(' '.join(detections)) + ".")

        cv2.imshow('Jetson Live Detection', img)

    def start(self):
        print ("Starting Live object detection, may take a few minutes to initialize...")
        self.camera.startStreaming()
        self.detector.initializeSession()

        if not self.camera.isOpened():
            print ("Camera has failed to open")
            exit(-1)
        elif self.debug:
            cv2.namedWindow("Jetson Live Detection", cv2.WINDOW_AUTOSIZE)
    
        while True:
            curr_time = time.time()

            img = self.camera.getFrame()
            scores, boxes, classes, num_detections = self.detector.detect(img)

            if self.debug:
                self._visualizeDetections(img, scores, boxes, classes, num_detections)
                print ("Debug: Running at: " + str(1.0/(time.time() - curr_time)) + " Hz.")

            if cv2.waitKey(1) == ord('q'):
                break
        
        cv2.destroyAllWindows()
        self.camera.__del__()
        self.detector.__del__()
        print ("Exiting...")
        return



if __name__ == "__main__":
    debug = True
    if len(sys.argv) > 2:
        debug = sys.argv[2]
        model = sys.argv[1]
    model = 'ssd_mobilenet_v1_coco_trt_graph.pb'
    live_detection = JetsonLiveObjectDetection(model=model, debug=debug)
    live_detection.start()
    

