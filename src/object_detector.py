import tensorflow as tf
import numpy as np
import cv2
import tensorflow.contrib.tensorrt as trt
import time
import sys

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

