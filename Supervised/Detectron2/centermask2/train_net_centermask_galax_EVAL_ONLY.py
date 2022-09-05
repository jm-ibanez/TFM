# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Youngwan Lee (ETRI), 2020. All Rights Reserved.
import logging
import os, json, cv2, random
from collections import OrderedDict
import torch
import numpy as np

import matplotlib.pyplot as plt
import random
import sys



import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch

from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.checkpoint import DetectionCheckpointer
from centermask.config import get_cfg

from detectron2 import model_zoo
from detectron2.structures import BoxMode

# from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation import inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor

from detectron2.utils.visualizer import Visualizer, ColorMode

output_dir = "/opt/TFM/output/"


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    
    cfg.DATASETS.TRAIN = ("galaxy_train",)
    cfg.DATASETS.TEST = ("galaxy_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.0000025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (galaxy). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.MODEL.DEVICE = 'cuda'

    # cfg.OUTPUT_DIR = output_dir

    cfg.merge_from_list(args.opts)

    ##cfg.freeze()
    default_setup(cfg, args)
    return cfg


def run_random_inference(cfg, dataset_dir, num_samples=3):
    """
    Inference & evaluation using the trained model
    Now, let's run inference with the trained model on the galaxy validation dataset. 
    First, let's create a predictor using the model we just trained:
    """

    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    
    confidence_threshold = 0.3

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold   # set a custom testing threshold
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = confidence_threshold
    #cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    
    predictor = DefaultPredictor(cfg)

    """
    Then, we randomly select several samples to visualize the prediction results.
    """

    dataset_dicts = get_galaxy_dicts(dataset_dir + "/val")
    galaxy_metadata = MetadataCatalog.get("galaxy_val")

    for d in random.sample(dataset_dicts, num_samples):    
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

        print("Decteced %d predictions" %len(outputs["instances"]))
        print(outputs["instances"])
        
        v = Visualizer(im[:, :, ::-1],
                       metadata=galaxy_metadata, 
                       scale=3.0, 
                       instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )

        # First, show the ground true
        out = v.draw_dataset_dict(d)
        plt.figure(figsize=(15,7))
        plt.imshow(out.get_image()[:, :, ::-1][..., ::-1])
        plt.show()
        
        # Now, show the predictions
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.figure(figsize=(15,7))
        plt.imshow(out.get_image()[:, :, ::-1])
        plt.show()

        # test
        # for box in outputs["instances"].pred_boxes.to('cpu'):
        #    v.draw_box(box)
        #    v.draw_text(str(box[:2].numpy()), tuple(box[:2].numpy()))
        #    v = v.get_output()
        #    img =  v.get_image()[:, :, ::-1]
        #    plt.imshow(img)
        #    plt.show()

    return cfg



def run_my_inference(cfg, dataset_dir, custom_files=None):
    """
    Inference & evaluation using the trained model
    Now, let's run inference with the trained model on the galaxy validation dataset. 
    First, let's create a predictor using the model we just trained:
    """

    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    confidence_threshold = 0.7   # set a custom testing threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold   # set a custom testing threshold
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = confidence_threshold
   
    predictor = DefaultPredictor(cfg)

    """
    Then, we select several samples from a given file to visualize the prediction results.
    """

    dataset_dicts = get_galaxy_dicts(dataset_dir + "/val")
    if custom_files!=None:
        my_files = list(np.loadtxt(dataset_dir + custom_files, dtype="str"))
        n_my_files = []
        for f in my_files:
            n_my_files.append(os.path.basename(f).split(".")[0])
        print(n_my_files)
    for d in random.sample(dataset_dicts, len(dataset_dicts)):
        f = os.path.basename(d["file_name"]).split(".")[0]
        if custom_files!=None and not f in n_my_files:
            continue
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                       metadata=galaxy_metadata, 
                       scale=3.0, 
                       instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )

        # First, show the ground true
        out = v.draw_dataset_dict(d)
        plt.figure(figsize=(15,7))
        plt.imshow(out.get_image()[:, :, ::-1][..., ::-1])
        # plt.show()
        
        # Now, show the prediction
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.figure(figsize=(15,7))
        plt.imshow(out.get_image()[:, :, ::-1])
        # plt.show()
        img = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_RGBA2RGB)
        plt.imsave(os.path.join(os.path.join(cfg.OUTPUT_DIR, 'visualization/centermask'), f + '.png'), img)

    return cfg

## OPC4
def show_random_train_sample(dataset_dir, num_samples=3):
    
    dataset_dicts = get_galaxy_dicts( dataset_dir + "/train")
    for d in random.sample(dataset_dicts, num_samples):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=galaxy_metadata, scale=3.0)
        out = visualizer.draw_dataset_dict(d)
        plt.figure(figsize=(15,7))
        plt.imshow(out.get_image()[:, :, ::-1][..., ::-1])
        plt.show()


def get_galaxy_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(list(imgs_anns.values())):
        record = {}
        
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []

        #print(annos)
        #print(type(annos))
        for anno in annos:
            #assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts

if __name__ == "__main__":

    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    dataset_dir = '/opt/TFM/DATASETS/new_subset/'
    # dataset_dir = '/home/ubuntu/JM/DATASETS/new_subset/'
    
    for d in ["train", "val"]:
        DatasetCatalog.register("galaxy_" + d, lambda d=d: get_galaxy_dicts(dataset_dir + d))
        MetadataCatalog.get("galaxy_" + d).set(thing_classes=["galaxy"], evaluator_type="coco")

    
    galaxy_metadata = MetadataCatalog.get("galaxy_train")

    cfg = setup(args)
    run_my_inference(cfg, dataset_dir, custom_files="my_files.txt")
    sys.exit(0)

