# Effect of label noise on  of object detection

This paper studies the sensitivity of object detection loss functions to label noise in bounding box detection tasks.
Although label noise has been widely studied in the classification context, less attention is paid to its effect on object detection. 
We characterize different types of label noise and concentrate on the most common type of annotation error, which is missing labels. We simulate missing labels by deliberately removing bounding boxes at training time and study its effect on different deep learning object detection architectures and their loss functions. Our primary focus is on comparing two particular loss functions: cross-entropy loss and focal loss. We also experiment on the effect of different focal loss hyperparameter values with varying amounts of noise in the datasets and discover that even up to 50% missing labels can be tolerated with an appropriate selection of hyperparameters. The results suggest that focal loss is more sensitive to label noise, but increasing the gamma value can boost its robustness.

## Datasets

In the paper, we used following three object detection dataset:
 * [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
 * [TUT Indoor](https://zenodo.org/record/2654485)
 * [FDDB face](http://vis-www.cs.umass.edu/fddb/)


## Experimental setup

- We use Cross entropy (CE) loss and Focal (FL) loss.
- We use Tensorflow Object detection API as our training platfrom. 
- We use single stage object detection, SSD MobileNet V1 network with MSCOCO pretained weight.

## Training
 
Training can be done as:
>python3 models/research/object_detection/model_main.py \
    	--pipeline_config_path=models/ssd_mobilenet_v1_coco_2018_01_28/pipeline.config  \
    	--train_dir=models/train

## References


* Other related works can be found [here](https://github.com/subeeshvasu/Awesome-Learning-with-Label-Noise). 