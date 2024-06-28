import os

import lightning as L

from model.hoax_detection_model import HoaxDetectionModel
from utils.preprocessor import Preprocessor

import argparse

def input_parser():
    parser = argparse.ArgumentParser()
        
    parser.add_argument("--use_gpu", action = "store_true")
    parser.add_argument("--max_epoch", type = int, default = 10)
    parser.add_argument("--batch_size", type = int, default = 10)
    parser.add_argument("--model_id", type = str, default = "indolem/indobert-base-uncased")
    parser.add_argument("--root_dir", type = str, default = "logs/indobert")
    
    args = parser.parse_args()
    
    if args.use_gpu:
        device = "gpu"
    else:
        device = "cpu"
        
    config = {
        "use_gpu": device,
        "max_epoch": args.max_epoch,
        "batch_size": args.batch_size,
        "model_id": args.model_id,
        "root_dir": args.root_dir
    }
    
    return config

if __name__ == "__main__":
    
    config = input_parser()
    
    dm = Preprocessor(
        batch_size = config["batch_size"]
    )
    
    model = HoaxDetectionModel(model_id = config["model_id"])
    
    trainer = L.Trainer(
        accelerator = config["use_gpu"],
        max_epochs = config["max_epoch"],
        default_root_dir = config["root_dir"]
    )
    
    # Ini bagian training 
    trainer.fit(model, datamodule = dm)
    # Testing model
    trainer.test(datamodel = dm, ckpt_path = 'best')
    