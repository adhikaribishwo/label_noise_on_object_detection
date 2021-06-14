# -*- coding: utf-8 -*-
"""
Usage : Convert annotation files from xml format to tfrecord format 
Run as: python3 convert_xml_tfrecord.py path_to_xml_files 
"""

import tensorflow as tf
from object_detection.utils import dataset_util
import cv2
import os
import sys
import xml.etree.ElementTree as ET
import glob


def create_tf_example(example):
      
    height = example["height"] # Image height
    width = example["width"] # Image width
    filename = example["filename"] # Filename of the image. Empty if image is not from file
    encoded_image_data = example["encoded_image_data"] # Encoded image bytes
    image_format = example["image_format"] # b'jpeg' or b'png'

    xmins = example["xmins"] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = example["xmaxs"]  # List of normalized right x coordinates in bounding box
             # (1 per box)
    ymins = example["ymins"]  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = example["ymaxs"] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
    
    classes_text = example["classes_text"] # List of string class name of bounding box (1 per box)
    classes = example["classes"]  # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def parse_xml(input_xml):

    examples = []
    path = os.path.dirname(input_xml) or "."
    
    tree = ET.parse(input_xml)
    root = tree.getroot()
    images = root.find('images')
    labels = []
    
    for idx, item in enumerate(images.getchildren()):
        
        example = {}
        
        filename = os.path.abspath(path + os.sep + item.get('file'))

        if filename.lower().endswith(('jpeg', 'jpg')):        
            example["image_format"] = b"jpeg"
        elif filename.lower().endswith('png'):
            example["image_format"] = b"png"
        else:
            print("Unknown file format: %s." % filename)
            continue
        
        example["filename"] = filename
        img = cv2.imread(filename)
        with open(filename, "rb") as fp:
            example["encoded_image_data"] = fp.read()
        
        example["filename"] = example["filename"].encode("utf-8")
        
        img_height, img_width, img_channels = img.shape

        example["height"]   = img_height
        example["width"]    = img_width
        example["channels"] = img_channels

        example["xmins"] = []
        example["xmaxs"] = []
        example["ymins"] = []
        example["ymaxs"] = []
        example["classes_text"] = []
        example["classes"] = []

        for i, box in enumerate(item.getchildren()):
            
            left   = float(box.get('left')) / img_width
            top    = float(box.get('top')) / img_height
            width  = float(box.get('width')) / img_width
            height = float(box.get('height')) / img_height
            right  = left + width
            bottom = top + height
            
            example["xmins"].append(left)
            example["ymins"].append(top)
            example["xmaxs"].append(right)
            example["ymaxs"].append(bottom)
            
            label = box.find("label").text
            if label not in labels:
                labels.append(label)
                print("[INFO] Found new label: %s" % label)
            class_idx = labels.index(label)

            example["classes_text"].append(label.encode('utf-8'))
            example["classes"].append(class_idx)
        
        examples.append(example)
        
        if idx % 100 == 0:
            print("[INFO]  Read image %d/%d..." % ((idx + 1), len(images.getchildren())))
        
    return examples

    
if __name__ == '__main__':

    if len(sys.argv) > 1:
        input_paths = [sys.argv[1]]
    else:
        input_paths  = glob.glob("*.xml")
        
    for input_path in input_paths:
        
        output_path = ".".join(input_path.split(".")[:-1] + ["record"])
    
        examples = parse_xml(input_path)
        
        writer = tf.python_io.TFRecordWriter(output_path)
    
        for idx, example in enumerate(examples):
            
            tf_example = create_tf_example(example)
            writer.write(tf_example.SerializeToString())
    
            print("[INFO] Wrote image %d/%d..." % ((idx + 1), len(examples)))
    
        writer.close()
        
        
        
