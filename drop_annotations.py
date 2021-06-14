# -*- coding: utf-8 -*-
"""
Usages: to drop a certain percentage of annotations (bounding boxes and labels) from the clean annotation file
Run as: python3  drop_annotations.py annotations_clean.xml annoation_noise_20.xml 0.2
"""

import sys
import numpy as np
import xml.etree.ElementTree as ET

if len(sys.argv) < 1 + 3:
    print("Give parameters: <original_file_name> <output_file_name> <percentage_of_dropped_boxes>")
    print("Example: python drop_anno.py train.xml train_d20.xml 0.2")
    sys.exit(1)
input_file_name = sys.argv[1]
output_file_name = sys.argv[2]
percentage = float(sys.argv[3])
assert percentage < 0.99999


print("[INFO] Reading {}...".format(input_file_name))
tree = ET.parse(input_file_name)
root = tree.getroot()
images = root.find('images')

print("[INFO] Dropping annotations...")
total_num = 0
for index, item in enumerate(images.getchildren()):
    total_num += len(item)
select_num = int(np.floor(total_num * (1.0-percentage)))
seq = np.arange(total_num)
seq_null_index = np.random.choice(seq, total_num - select_num, replace=False)
seq[seq_null_index] = -1
### check
unique, counts = np.unique(seq, return_counts=True)
uq = dict(zip(unique,counts))
#print("{} vs {} vs {}".format(select_num, uq.get(-1), len(seq_null_index)))
#print("{}".format(seq_null_index))
assert(total_num - select_num == uq.get(-1))
### /check
box_index = 0
for index,item in reversed(list(enumerate(images.getchildren()))):
    for bindex, bitem in reversed(list(enumerate(item.getchildren()))):
        #print("{}".format(bindex))
        if seq[box_index] == -1:
            print("-- From file {}: dropped box {},{},{},{}".format(item.get("file"), bitem.get("left"), bitem.get("top"), bitem.get("width"), bitem.get("height")))
            #import pdb; pdb.set_trace()
            item.remove(bitem)
        box_index += 1
    
print("Dropped {}% annotations. Went from {} to {}.".format(percentage*100, total_num, select_num))

new_total = 0
for index,item in enumerate(images.getchildren()):
    new_total += len(item)
assert new_total == select_num

print("[INFO] Writing {}...".format(output_file_name))
tree.write(output_file_name)

print("[INFO] All done!")
