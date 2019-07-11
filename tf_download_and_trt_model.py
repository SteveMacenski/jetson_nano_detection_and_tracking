import tensorflow.contrib.tensorrt as trt
import sys
import os
try:
  from tf_trt_models.detection import download_detection_model, build_detection_graph
except:
  from tf_trt_models.tf_trt_models.detection import download_detection_model, build_detection_graph

# Options in model zoo: ssd_inception_v2_coco, ssd_mobilenet_v2_coco, ssd_mobilenet_v1_coco, ssdlite_mobilenet_v2_coco, ssd_mobilenet_v2_quantized_coco

# Options in Nvidia TRT Downloader: ssd_inception_v2_coco, ssd_mobilenet_v1_coco, ssd_mobilenet_v2_coco, ssd_resnet_50_fpn_coco, faster_rcnn_resnet50_coco, faster_rcnn_nas, mask_rcnn_resnet50_atrous_coco at 300x300 

MODEL = 'ssd_mobilenet_v1_coco' if len(sys.argv) < 2 else sys.argv[1]

print ("Downloading model " + MODEL + "..." )
config_path, checkpoint_path = download_detection_model(MODEL, './data')

print ("Building detection graph from model " + MODEL + "...")
frozen_graph, input_names, output_names = build_detection_graph(
    config=config_path,
    checkpoint=checkpoint_path,
    score_threshold=0.3,
    #iou_threshold=0.5,
    batch_size=1
)

# download the detection model and then build the graph locally
# score_threshold is the score below to throw out BBs
# iou is the intersect over union ratio for non-max supression
# batch_size is 1 for the Nano for speed


print ("Creating Jetson optimized graph...")
trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph,
    outputs=output_names,
    max_batch_size=1,
    max_workspace_size_bytes=1 << 25,
    precision_mode='FP16',
    minimum_segment_size=50
)

# make the graph a trt for Jetson optimizations
# precision mode is the most important for the Nano's architecture

print ("Saving trt optimized graph...")

with open('./data/' + MODEL + '_trt_graph.pb', 'wb') as f:
    f.write(trt_graph.SerializeToString())

print ("Done! Have a great day :-)")
