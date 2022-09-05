import logging

from detectron2.data import MetadataCatalog, DatasetCatalog
#from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch



from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.structures import BoxMode

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader



from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
import matplotlib.pyplot as plt
import cv2
import os
import json
import numpy as np
import random
import sys


from detectron2.utils.logger import setup_logger

# Main directories
#dataset_dir = '/opt/TFM/DATASETS/tutorial/'

# LOCAL
dataset_dir = '/opt/TFM/DATASETS/new_subset/'
output_dir = "/opt/TFM/output/"

# LAMBDALABS
# dataset_dir = '/home/ubuntu/JM/DATASETS/new_subset/'
# output_dir = "/home/ubuntu/JM/output/"

log_file = "mrncc_galaxy.log"
logger = setup_logger(output=output_dir + log_file)



def custom_config(num_classes, args):
    
    cfg = get_cfg() 
    # get configuration from model_zoo
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # Model (in fact, next are not needed because they have already been setup in .yaml config file)
    cfg.MODEL.MASK_ON = True
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes # only has one class (galaxy)
    cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
    #cfg.MODEL.RESNETS.DEPTH = 34
    #cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 64
    
    # Solver
    cfg.SOLVER.BASE_LR = 0.0002
    cfg.SOLVER.MAX_ITER = 400
    cfg.SOLVER.STEPS = (20, 10000, 20000)
    cfg.SOLVER.gamma = 0.5
    cfg.SOLVER.IMS_PER_BATCH = 16
    
    # Test
    cfg.TEST.DETECTIONS_PER_IMAGE = 20
    
    # INPUT
    cfg.INPUT.MIN_SIZE_TRAIN = (800,)
    
    # DATASETS
    cfg.DATASETS.TEST = ('galaxy_val',)
    cfg.DATASETS.TRAIN = ('galaxy_train',)
    
    # DATASETS
    cfg.OUTPUT_DIR = output_dir
    
    # Computing
    # cfg.MODEL.DEVICE = 'cpu'

    # from colab demo
    #cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    # end colab demo

    
    #cfg.freeze()
    default_setup(cfg, args)

    return cfg

###########  VISUALIZATION STUFF ##################################################
# OPC1
def visualization(metadata, cfg, test_set):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    predictor = DefaultPredictor(cfg)
    for d in random.sample(test_set, 3):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=metadata,
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE_BW
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        img = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_RGBA2RGB)
        plt.imsave(os.path.join(os.path.join(cfg.OUTPUT_DIR, 'visualization'), str(d["image_id"]) + '.png'), img)

# OPC2 -- for tensorflow
def random_visualization(cfg, dataset_metadata, dataset_dicts, num_samples=3):

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    predictor = DefaultPredictor(cfg)
    
    for d in random.sample(dataset_dicts, num_samples):    
        im = cv2.imread(d["file_name"])
        print("FILENAME=", d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=dataset_metadata, 
                       scale=0.8, 
                       instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2_imshow(v.get_image()[:, :, ::-1]) ## only for tensorflow notebooks !! 

# OPC3 -- prefered
def run_random_inference(cfg, dataset_dir, num_samples=3, custom_files=None):
    """
    Inference & evaluation using the trained model
    Now, let's run inference with the trained model on the galaxy validation dataset. 
    First, let's create a predictor using the model we just trained:
    """

    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    """
    Then, we randomly select several samples to visualize the prediction results.
    """

    dataset_dicts = get_galaxy_dicts(dataset_dir + "val")
    if custom_files!=None:
        my_files = list(np.loadtxt(dataset_dir + custom_files, dtype="str"))
        n_my_files = []
        for f in my_files:
            n_my_files.append(os.path.basename(f).split(".")[0])
        print(n_my_files)
    for d in random.sample(dataset_dicts, len(dataset_dicts)): #num_samples):
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
        plt.imsave(os.path.join(os.path.join(cfg.OUTPUT_DIR, 'visualization/MRCNN'), f + '.png'), img)

        # test
        # for box in outputs["instances"].pred_boxes.to('cpu'):
        #    v.draw_box(box)
        #    v.draw_text(str(box[:2].numpy()), tuple(box[:2].numpy()))
        #    v = v.get_output()
        #    img =  v.get_image()[:, :, ::-1]
        #    plt.imshow(img)
        #    plt.show()

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


## Load dataset (JSON based)
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


def load_dataset():
    
    for d in ["train", "val"]:
        DatasetCatalog.register("galaxy_" + d, lambda d=d: get_galaxy_dicts("tutorial/" + d))
        MetadataCatalog.get("galaxy_" + d).set(thing_classes=["galaxy"])
    galaxy_metadata = MetadataCatalog.get("galaxy_train")



def run_metric_performances(cfg):
    """
    We can also evaluate its performance using AP metric implemented in COCO API.
    This gives an AP of ~70. Not bad!
    """

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("galaxy_val", output_dir=cfg.OUTPUT_DIR + "/output")
    val_loader = build_detection_test_loader(cfg, "galaxy_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))
    # another equivalent way to evaluate the model is to use `trainer.test`


# ------------------------------------------------------------------------------
 
if __name__ == '__main__':
  
  logger.info("***** Starting...*****")

  args = default_argument_parser().parse_args()
  print("Command Line Args:", args)


  for d in ["train", "val"]:
    DatasetCatalog.register("galaxy_" + d, lambda d=d: get_galaxy_dicts(dataset_dir + d))
    MetadataCatalog.get("galaxy_" + d).set(thing_classes=["galaxy"])
  
  logger.info("DATASETS leidos !!")
  galaxy_metadata = MetadataCatalog.get("galaxy_train")

  cfg = custom_config(num_classes=1, args=args)

  logger.info("Setting up trainer...")

  print(args)

  os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

  if args.eval_only:
    # show_random_train_sample(dataset_dir)
    run_random_inference(cfg, dataset_dir, custom_files="my_files.txt")
    sys.exit(0)


  trainer = DefaultTrainer(cfg)
  trainer.resume_or_load(resume=False)
  trainer.train()


  run_metric_performances(cfg)