import tensorflow as tf
import numpy as np
import cv2
import tensorflow.contrib.tensorrt as trt
import time
import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst



""" TensorFlow detection using TRT optimized graph"""
class ObjectDetection():
    def __init__(self, detection_model_path = './data/ssd_mobilenet_v2_coco_trt_graph.pb'):
        self.detection_model_path = detection_model_path
        self.labels = self._getLabels()

    def _getLabels(self):
        labels = {}
        with open('./data/coco_classes.json') as fh:
            for line in fh:
                label, des = line.strip().split(': ', 1)
                labels[label] = des.strip()
        return labels

    def detect(self, frame):
        img_expanded = np.expand_dims(frame, axis=0)
        scores, boxes, classes, num_detections = self.tf_sess.run(self.tf_tensors, feed_dict={self.tf_input: img_expanded})
        return scores[0], boxes[0], classes[0], int(num_detections[0])

    def _setupTensors(self):
        self.tf_input = self.tf_sess.graph.get_tensor_by_name('image_tensor:0')
        tf_scores = self.tf_sess.graph.get_tensor_by_name('detection_scores:0')
        tf_boxes = self.tf_sess.graph.get_tensor_by_name('detection_boxes:0')
        tf_classes = self.tf_sess.graph.get_tensor_by_name('detection_classes:0')
        tf_num_detections = self.tf_sess.graph.get_tensor_by_name('num_detections:0')
        self.tf_tensors = [tf_scores, tf_boxes, tf_classes, tf_num_detections]

    def _getTRTGraph(self):
        with tf.gfile.FastGFile(self.detection_model_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        return graph_def

    def initializeSession(self):
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.tf_sess = tf.Session(config=tf_config)
        tf.import_graph_def(self._getTRTGraph(), name='')
        self._setupTensors()
        print ("Successfully initialized TF session")

    def __del__(self):
        tf.reset_default_graph()
        self.tf_sess.close()
        print ("Cleanly exited ObjectDetector")



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



""" Jetson Live Object Detector """
class JetsonLiveObjectDetection():
    def __init__(self, model,  debug=False):
        self.debug = debug
        self.camera = MipiCamera(300, 300)
        self.model = model
        self.detector = ObjectDetection('./data/' + self.model)

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

    def _publishDetections(self, curr_time, scores, boxes, classes, num_detections):
        #msg = 2DDetections()
        #msg.header.stamp = curr_time
        #msg.header.frame_id = "mipi_camera"
        #for i in range(num_detections):
        #    det = 2DDetection()
        #    det.object = self.detector.labels[str(int(classes[i]))]
        #    det.score = float(scores[i])
        #    det.corners = (bbox[1]*cols, bbox[0]*rows, bbox[3]*cols, bbox[1]*rows)
        #    msg.detections.push_back(det)
        #self.detectionPub.publish(msg)
        pass

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
            self._publishDetections(curr_time, scores, boxes, classes, num_detections)

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
    

