import os
import time
import glob

import numpy as np
from matplotlib import pyplot as plt

from detectron2.engine import DefaultPredictor, default_argument_parser, default_setup, launch
from centermask.config import get_cfg

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    
    outputs_cpu = []
    predictor = DefaultPredictor(cfg)
    for file_name in glob.glob("./images/*"):
        im = plt.imread(file_name)        
        
        start_time = time.time()
        outputs = predictor(im)
        output_cpu = outputs["instances"].to("cpu")
        print(f"[TIME] {time.time() - start_time}s")
        for i, mask in enumerate(np.array(output_cpu.get_fields()["pred_masks"])):
            plt.imsave(f"./masks/{os.path.basename(file_name)}_mask_{i}.png", mask*255)
        outputs_cpu.append(output_cpu)
    return output_cpu


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    
    # Regist own dataset.
    from detectron2.data.datasets import register_coco_instances
    # train data
    name        = "numberplate_train"
    json_file   = "/var/www/centermask2/datasets/numberplate/train/coco_numberplate.json"
    image_root  = "/var/www/centermask2/datasets/numberplate/train"
    # test data
    name_val        = "numberplate_val"
    json_file_val   = "/var/www/centermask2/datasets/numberplate/val/coco_numberplate.json"
    image_root_val  = "/var/www/centermask2/datasets/numberplate/val"
    # registr
    register_coco_instances(name, {}, json_file, image_root)
    register_coco_instances(name_val, {}, json_file_val, image_root_val)
    
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


