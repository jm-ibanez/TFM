# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Youngwan Lee (ETRI), 2020. All Rights Reserved.
import logging
import os, json, cv2, random
from collections import OrderedDict
import torch
import numpy as np

import random
import sys



import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    # CityscapesInstanceEvaluator,
    # CityscapesSemSegEvaluator,
    # COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from centermask.evaluation import (
    COCOEvaluator,
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.checkpoint import DetectionCheckpointer
from centermask.config import get_cfg

from detectron2 import model_zoo
from detectron2.structures import BoxMode

# from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation import inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor



class Trainer(DefaultTrainer):
    """
    This is the same Trainer except that we rewrite the
    `build_train_loader` method.
    """



    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res



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


    cfg.merge_from_list(args.opts)

    ##cfg.freeze()
    default_setup(cfg, args)
    return cfg


def run_metric_performances(cfg):
    """
    We can also evaluate its performance using AP metric implemented in COCO API.
    This gives an AP of ~70. Not bad!
    """

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("galaxy_val", output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "galaxy_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))
    # another equivalent way to evaluate the model is to use `trainer.test`


def run_random_inference(cfg, dataset_dir, num_samples=3):
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

    dataset_dicts = get_galaxy_dicts(dataset_dir + "/val")
    for d in random.sample(dataset_dicts, num_samples):    
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
        plt.show()
        
        # Now, show the prediction
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

def main(args):

    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )


    r = trainer.train()

    # run_metric_performances(cfg)

    return r

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

    dataset_dir = '/opt/TFM/DEVELOP/datasets/tutorial/'
    # dataset_dir = '/home/ubuntu/JM/DATASETS/new_subset/'
    
    for d in ["train", "val"]:
        DatasetCatalog.register("galaxy_" + d, lambda d=d: get_galaxy_dicts(dataset_dir + d))
        MetadataCatalog.get("galaxy_" + d).set(thing_classes=["galaxy"], evaluator_type="coco")

    
    if args.eval_only:
        cfg = setup(args)
        run_random_inference(cfg, dataset_dir)
        sys.exit(0)


    #galaxy_metadata = MetadataCatalog.get("galaxy_train")

    #launch(
    #    main,
    #    args.num_gpus,
    #    num_machines=args.num_machines,
    #    machine_rank=args.machine_rank,
    #    dist_url=args.dist_url,
    #    args=(args,),
    #)

    # Multi-GPU training
    launch(
        main,
        args.num_gpus,  # Number of GPUs per machine
        num_machines=1,
        machine_rank=0,
        dist_url="tcp://127.0.0.1:1234",
        args=(args,),
    )
