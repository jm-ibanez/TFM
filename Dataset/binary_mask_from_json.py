"""
Convert a mask definded in JSON file with VGG annotation format 
into binary label image""
"""

import json
import os
import numpy as np
import PIL.Image
import cv2
import matplotlib.pyplot as plt

import pandas as pd

# gz2_table = pd.read_csv('http://gz2hart.s3.amazonaws.com/gz2_hart16.csv.gz', compression='gzip')
# gz2_table = pd.read_csv('https://zooniverse-data.s3.amazonaws.com/galaxy-zoo-2/zoo2MainSpecz.csv.gz',compression='gzip')

gz2_file = '/opt/TFM/DATASETS/GZ2/zoo2MainSpecz.csv'

gz2_table = pd.read_csv(gz2_file)

# Reduce galaxy classes to 3 classes (S-piral, E-lliptical, A-rtifact (or star))
gz2_table['simple_class'] = gz2_table['gz2class'].apply(lambda x: x[0])


def binary_mask_from_annot(json_input_filename, dataset_dir, seg_map_suffix="_mask"):

    # Read JSON file
    with open(json_input_filename, "r") as read_file:
        data = json.load(read_file)

    # All image filenames in JSON file
    all_file_names = list(data.keys())
    print(all_file_names)

    # Look for image filenames in dataset_dir
    files_in_directory = []
    for root, dirs, files in os.walk(dataset_dir):
        for filename in files:
            files_in_directory.append(filename)

    # Create directories
    ann_path = os.path.join(dataset_dir, "annotations")
    img_path = os.path.join(dataset_dir, "images")
    os.makedirs(ann_path, exist_ok = True)
    os.makedirs(img_path, exist_ok = True)

    spiral = 0
    ellipical = 0
    unknown = 0                
    
    for j in range(len(all_file_names)): 
        image_name = data[all_file_names[j]]['filename']
        # checks filenames that match 
        if image_name in files_in_directory: 
            print("MATCHED FILENAME= %s" %(dataset_dir + image_name))
            img = np.asarray(PIL.Image.open(dataset_dir + image_name))    
        else:
            # json filename does not exist in dataset_dir  
            continue

        # Look for galaxy type
        id = int(os.path.basename(image_name).split(".")[0].split("_")[1])
        print("ID", id)
        gal = gz2_table[gz2_table['dr7objid'] == id]
        #print("VALUES -->", gal['simple_class'].values)
        if len(gal) > 0:
            if gal['simple_class'].values[0] == 'E':
                gtype = 'E'
                ellipical += 1
            elif gal['simple_class'].values[0] == 'S':
                gtype = 'S'
                spiral += 1
            else:
                gtype = 'A'
                print("File with Star or Artifacts--> ", f)
                unknown +=1
        else:
            print("dr7objid not found. Assume Artifacts.")
            gtype = 'A'

        if data[all_file_names[j]]['regions'] != {}:
            try:
                filename_wo_ext = image_name.split('.')[0]
                cv2.imwrite(img_path + '/%s' % filename_wo_ext +'.jpg', img)
                #cv2.imwrite('images/%05.0f' % j +'.jpg', img)
                print(j)
                try: 
                     shape1_x = data[all_file_names[j]]['regions']['0']['shape_attributes']['all_points_x']
                     shape1_y = data[all_file_names[j]]['regions']['0']['shape_attributes']['all_points_y']
                except : 
                     shape1_x = data[all_file_names[j]]['regions'][0]['shape_attributes']['all_points_x']
                     shape1_y = data[all_file_names[j]]['regions'][0]['shape_attributes']['all_points_y']
            
                #print("Vamos a mostrar...:",shape1_x)
                
                #fig = plt.figure()
                #plt.imshow(img.astype(np.uint8)) 
                #plt.scatter(shape1_x, shape1_y, zorder=2, color='red', marker = '.', s=55)
                

                ab = np.stack((shape1_x, shape1_y), axis=1)
                
                img2 = cv2.drawContours(img, [ab], -1, (255,255,255), -1)
               
                # https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html
                # category_id: an integer in the range [0, num_categories-1] representing the category label. 
                # The value num_categories is reserved to represent the “background” category, if applicable
                # In our case, then 0 is the label of the background
                mask = np.zeros((img.shape[0], img.shape[1]))
                
                #print("IM_SHAPE: %d x %d" %(img.shape[0], img.shape[1]))
                #print(ab)
                if gtype == 'E':
                    CLASS_VALUE = 1
                elif gtype == 'S':
                    CLASS_VALUE = 2
                else:
                    CLASS_VALUE = 255 # ignore_value

                img3 = cv2.drawContours(mask, [ab], -1, CLASS_VALUE, -1)
                
                cv2.imwrite(ann_path + '/%s' % filename_wo_ext + seg_map_suffix + '.png', mask.astype(np.uint8))

            except Exception as e:
                print(e)

# ------------------------------------------------------------------------------

if __name__ == '__main__':

    JSON_FILENAME = "via_region_data.json"
    
    DATASETDIR_TRAIN = "/home/ubuntu/JM/DATASETS/tutorial_2/train/"
    DATASETDIR_VAL = "/home/ubuntu/JM/DATASETS/tutorial_2/val/"
    
    seg_map_suffix = '_mask'
    
    # Train
    print("TRAIN dataset conversion ...")
    try:
        binary_mask_from_annot(DATASETDIR_TRAIN + JSON_FILENAME, DATASETDIR_TRAIN, seg_map_suffix)
    except Exception as e:
        raise e    


    print("VAL dataset conversion ...")
    try:
        binary_mask_from_annot(FILENAME_VAL, DATASETDIR_VAL, seg_map_suffix)
    except Exception as e:
        raise e
    