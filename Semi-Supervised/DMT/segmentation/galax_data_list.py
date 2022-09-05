import os
from utils.common import base_galax as base
import random

def traverse(images_dir, data_list):
    for item in sorted(os.listdir(images_dir)):
        dir_path = os.path.join(images_dir, item)
        if os.path.isdir(dir_path):
            for image in sorted(os.listdir(dir_path)):
                temp = item + '/' + image.split('.jpg')[0] + '\n'
                data_list.append(temp)
        else:
            # is a file
            temp = item.split('.jpg')[0] + '\n'
            data_list.append(temp)


# Traverse images
train_list = []
val_list = []
test_list = []


# Get the Whole files
traverse(os.path.join(base, "images/train"), train_list)
traverse(os.path.join(base, "images/val"), val_list)
#traverse(os.path.join(base, "images/test"), test_list)
print('Whole Training set size: ' + str(len(train_list)))
print('Whole Validation set size: ' + str(len(val_list)))
print('Whole Test set size: ' + str(len(test_list)))

# Random Sample and 10% of dataset samples
random.seed(123456)
train_list = random.sample(train_list, int(len(train_list)*0.1))
val_list = random.sample(val_list, int(len(val_list)*0.1))
print('New SAMPLE of Training set size: ' + str(len(train_list)))
print('New SAMPLE of Validation set size: ' + str(len(val_list)))
print('New SAMPLE of Test set size: ' + str(len(test_list)))

# Save training list
lists_dir = os.path.join(base, "data_lists_small_10")
if not os.path.exists(lists_dir):
    os.makedirs(lists_dir)
with open(os.path.join(lists_dir, "train.txt"), "w") as f:
    f.writelines(train_list)
with open(os.path.join(lists_dir, "val.txt"), "w") as f:
    f.writelines(val_list)
with open(os.path.join(lists_dir, "test.txt"), "w") as f:
    f.writelines(test_list)

print("New file list in: ", lists_dir)
print("Complete.")
