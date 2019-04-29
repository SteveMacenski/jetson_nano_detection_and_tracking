echo "Clone this repo and place it where you would like to have all the detection/model API installed and run this script in the root of that directory."

echo "Install jetson nano swap memory of 6GB. Reboot to take effect"
./install_scripts/install_nano_swap.sh

echo "Install Tensorflow and dependencies"
./install_scripts/install_deps.sh

echo "Install TF models and detection API"
./install_scripts/install_tf_models.sh

sudo apt install curl
curl https://bootstrap.pypa.io/get-pip.py | sudo python3

echo "done!"

echo "You may now install pretrained models via:"
echo "python3 tf_download_and_trt_model.py ssd_mobilenet_v1_coco / ssd_mobilenet_v2_coco / etc"
echo ""
echo "Which will install in ./data and the TRT model will be saved as [name]_trt_graph.pb."
echo ""
echo "Then utilize by changing the model to use in the JetsonObjectDetection program via:"
echo "python3 jetson_live_object_detection.py [model]_trt_graph.pb True"
echo "Where True/False is debug for printing statements and showing visualizations."
echo "This should also be run in the local directory for the time being until absolute paths are used."
echo ""
echo "Good luck!"
