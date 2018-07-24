# CarND-Capstone-TrafficLightDetection

1. conda create -n tensorflow1.8 python=3.5 && conda install tensorflow-gpu=1.8.0

2. ```
   sudo apt-get install protobuf-compiler
   sudo pip install pillow
   sudo pip install lxml
   sudo pip install jupyter
   sudo pip install matplotlib
   ```
3. 	export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

4. 	Go to root directory,  get https://github.com/tensorflow/models/tree/master/research/object_detection  as object_detection

5. 	Get annoted img data https://drive.google.com/file/d/0B-Eiyn-CUQtxdUZWMkFfQzdObUE/view?usp=sharing and unzip to data/real_training_data,data/sim_training_data

6. 	Get http://storage.googleapis.com/download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz and unzip to data/

7. 	data/config/faster_rcnn_resnet101_udacitycapstonejunior.config

8. 	```
	python data_conversion_udacity_sim.py --output_path data/sim_data.record
	python data_conversion_udacity_real.py --output_path data/real_data.record
	```

9.	```
	mkdir model_frozen_sim; 
	mkdir model_frozen_real; 
	```
	
10. 	python object_detection/train.py --pipeline_config_path=data/config/faster_rcnn_resnet101_udacitycapstonejunior.config --train_dir=data/sim_training_data/sim_data_capture

11. python object_detection/export_inference_graph.py --pipeline_config_path=config/faster_rcnn-traffic-udacity_sim.config --trained_checkpoint_prefix=data/sim_training_data/sim_data_capture/model.ckpt-5000 --output_directory=model_frozen_sim/ --input_type image_tensor

12.	python object_detection/train.py --pipeline_config_path=data/config/faster_rcnn_resnet101_udacitycapstonejunior.config --train_dir=data/real_training_data/real_data_capture

13.	python object_detection_ori/export_inference_graph.py --pipeline_config_path=data/config/faster_rcnn_resnet101_udacitycapstonejunior.config --trained_checkpoint_prefix=data/real_training_data/real_data_capture/model.ckpt-21516 --output_directory=model_frozen_real/ --input_type image_tensor
