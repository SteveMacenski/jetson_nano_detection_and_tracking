echo "Installing depedencies. If this is being run on a Nano, might take a while..."
sudo apt-get install libfreetype6-dev protobuf-compiler python-pil python-lxml python-tk
sudo pip3 install Cython contextlib2 pillow lxml matplotlib
sudo pip3 install pycocotools

echo "Installing TF TRT Models"
git clone --recursive https://github.com/NVIDIA-Jetson/tf_trt_models.git
cd tf_trt_models

sudo ./install.sh python3

echo "Testing model install..."
cd third_party/models/research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
python3 object_detection/builders/model_builder_test.py

echo 'export PYTHONPATH=$PYTHONPATH:'`pwd`':'`pwd`'/slim' >> $HOME/.bashrc

echo "Done!"

