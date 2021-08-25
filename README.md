# Effect of label noise on object detection

This paper studies the sensitivity of object detection loss functions to label noise in bounding box detection tasks.
Although label noise has been widely studied in the classification context, less attention is paid to its effect on object detection. 
We characterize different types of label noise and concentrate on the most common type of annotation error, which is missing labels. We simulate missing labels by deliberately removing bounding boxes at training time and study its effect on different deep learning object detection architectures and their loss functions. Our primary focus is on comparing two particular loss functions: cross-entropy loss and focal loss. We also experiment on the effect of different focal loss hyperparameter values with varying amounts of noise in the datasets and discover that even up to 50% missing labels can be tolerated with an appropriate selection of hyperparameters. The results suggest that focal loss is more sensitive to label noise, but increasing the gamma value can boost its robustness.

## Datasets

In the paper, we used following three object detection dataset:
 * [PASCAL VOC 2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
 * [Indoor dataset](https://zenodo.org/record/2654485)
 * [FDDB face dataset](http://vis-www.cs.umass.edu/fddb/)


## Scripts
- *drop_annotations.py* removes the desire amount of noise and create noisly annoations(in xml format)
- *convert_xml_tfrecord.py*  converts the annoation from xml format to tensorflow record (tfrecord) format


## Experimental setup

- We use Tensorflow Object detection API as our training platfrom. Installation guide can be found [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/tensorflow-1.14/install.html).
- We use single stage object detection, SSD MobileNet V1 network with MSCOCO pretained weight downloaded from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md).
 
  
- On working directory 
  -   Download tensorflow models repository in **models**
  -   Download pretrained weight in **pretrained_weight**  


   **Structure of working direcotry**

    ```
    project_directory
    │   README.md
    │   convert_xml_tfrecord.py 
    |   drop_annotations.py
    |   
    │
    └─── models
    │   │   ...
    │   │
    │   └─── Research
    │       │ ...
    |       └─── Object detection
    |             └─── ...
    |                 └─── ...
    │   
    └─── pretrained_weight
    |    └───  ssd_mobilenet_v1_coco_2018_01_28
    │            │   CE_pipeline.config
    │            │   FL_pipeline.config
    │            │   checkpoint
    │            │   ...
    |            └─── saved_model
    |
    └─── trained_models
    |    | ...
    │   
    └─── data
        └─── Indoor
        └─── PASCAL VOC
        └─── FDDB
        └─── Results
            | plot_results.py
            | graphs*.pdf


    ```


- Training can be done as:

    >python3 models/research/object_detection/model_main.py  --pipeline_config_path=pretrained_weight/ssd_mobilenet_v1_coco_2018_01_28/CE_pipeline.config  --train_dir=trained_models/train

    Or 

    >python3 models/research/object_detection/model_main.py --pipeline_config_path=pretrained_weight/ssd_mobilenet_v1_coco_2018_01_28/CE_pipeline.config --train_dir=trained_models/train



## Reference

If you use this work, please cite as follow

   @InProceedings{adhikari_2021LN,
   author="Adhikari, Bishwo and Peltom{\"a}ki, Jukka and Germi, Saeed Bakhshi and Rahtu, Esa and Huttunen, Heikki",
   editor="Habli, Ibrahim and Sujan, Mark and Gerasimou, Simos and Schoitsch, Erwin and Bitsch, Friedemann",
   title="Effect of Label Noise on Robustness of Deep Neural Network Object Detectors",
   booktitle="Computer Safety, Reliability, and Security. SAFECOMP 2021 Workshops",
   year="2021",
   publisher="Springer International Publishing",
   address="Cham",
   pages="239--250",
   isbn="978-3-030-83906-2"}

    


* Other related works can be found [here](https://github.com/subeeshvasu/Awesome-Learning-with-Label-Noise). 
