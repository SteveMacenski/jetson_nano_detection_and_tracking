# jetson_nano_detection_and_tracking
Jetson Nano ML install scripts, automated optimization of robotics detection models, and filter-based tracking of detections

## Motivation

Installing and setting up the new Nvidia Jetson Nano was surprisingly time consuming and unintuitive. From protobuf version conflicts, to Tensorflow versions, OpenCV recompiling with GPU, models running, models optimized, and general chaos in the ranks.

This repository is my set of install tools to get the Nano up and running with a convincing and scalable demo for robot-centric uses. In particular, using detection and semantic segmentation models capable at running in real-time on a robot for $100. By convincing, I mean not using Nvidia's 2-day startup model you just compile and have magically working without having control. This gives you full control of which model to run and when. 

In the repository, you'll find a few key things:

### Install of dependencies

Getting the right versions of Tensorflow, protobufs, etc and having everyone play well on the Jetson Nano platform was a big hassle. Hopefully these will help you.

### Download of pretrained models for real-time detection 

Scripts to automatically download pretrained Tensorflow inference graphs and checkpoints, then optimize with TensorRT (which I found as a critical must-have to even *run* on the Nano).

Also there's nothing here that prohibits you from using your own Tensorflow and then using the same scripts to optimize it with TensorRT and then deploy as described below. I have retrained a model from the zoo and followed these same instructions with equal success.

### Execution of live detection with an attached MIPI-based camera

This will run the argus streamer for a MIPI camera compatible with the Jetson Nano. There are a number out there available, I happen to use the Raspberry Pi v2.1 camera simply because I had it around from another project and also because its shockingly high resolution for a $20 toy. 

### Filter-based tracking of detections

This uses a constant velocity Kalmon Filter to track detections in the image frame and report stabilized detections based on the centroid. This is to handle 2 things. The first is to deal with irregular detections so that a few missing frames doesn't make an upstream application think a person disppeared out of thin air for 57 ms. Secondarily, it acts as smoother so if individual frames detect irraneous things (like an airplane rather than my ear) single frame detections aren't introduced into the system. For robotics applications it would be pretty bad if we saw an airplane in my living room. 
