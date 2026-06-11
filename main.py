from train import train_loop, train_loop_one_image
from load_dataset import create_dataloader, create_dataloader_one_image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import logging
import os
import re
from PIL import ImageFile
from utils import set_seed, get_model

ImageFile.LOAD_TRUNCATED_IMAGES = True

def main(args):
    
    path_save = os.path.join(args.save_dir, args.model_naming)
    os.makedirs(path_save, exist_ok=True)

    log_file = os.path.join(path_save, "logger.log")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[
                            logging.FileHandler(log_file, mode="w"),
                            logging.StreamHandler()  # Enables logging to console
                        ])
    logger = logging.getLogger()
    
    writer = SummaryWriter(log_dir=os.path.abspath(path_save))

    logger.info("Starting Training Pipeline")
    set_seed(args.seed)
    logger.info(f"Set seed to {args.seed}")
    logger.info(f"Using JSON: {args.json_fold}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device used: {device}")
    logger.info("Start loading data")


    if "siamese" in args.model_config.lower():
        train_loader, val_loader, test_loader = create_dataloader(args.batch_size,
                                                                args.json_fold,
                                                                args.sampler,
                                                                args.data_setup,
                                                                logger)
    else:
        train_loader, val_loader, test_loader = create_dataloader_one_image(args.batch_size,
                                                            args.json_fold,
                                                            args.sampler,
                                                            args.data_setup,
                                                            logger)
    logger.info('Finished loading data')

    logger.info(f"Starting model creation with model: {args.model_config}")


    if "linprob" in args.model_config.lower():
        logger.info(f"Layer for LinProbe: {args.layer}")
        num = str(re.search(r'fold_(\d+)\.json', args.json_fold).group(1))
        temp_lin = f"/lab/deasylab1/Jorge/Endo_2025_Temporal/finetunedmodel_allSameCrop_115/MainResults/Siam-SwinSmall_Fold{num}_TemporalChange_115/model_epoch30.pt"
        logger.info(f"Loading weights: {temp_lin}")
        model = get_model(args.model_config, args.layer, args.num_classes)
        model = model.to(device)
        if "SDCCA" in args.model_config.lower():
            x = f"/lab/deasylab1/Jorge/Endo_2025_Temporal/finetunedmodel_allSameCrop_115/MainResults/Siam-SwinS_CrossDual_Fold{num}_TemporalChange_115/model_epoch30.pt"
        else:
            x = f"/lab/deasylab1/Jorge/Endo_2025_Temporal/finetunedmodel_allSameCrop_115/MainResults/Siam-SwinSmall_Fold{num}_TemporalChange_115/model_epoch30.pt"
        ckpt = torch.load(x, map_location='cpu')
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        filtered = {k: v for k, v in state_dict.items() if not k.startswith("head")}
        model.load_state_dict(filtered, strict=False) 
    else:
        print("Training from ImageNet Weights")
        model = get_model(args.model_config, args.layer, args.num_classes)
        model = model.to(device)


    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info("Total Number of Parameters: " + str(pytorch_total_params))
    logger.info("Finished model creation")

    logger.info("Defining Loss function and Optimizer")
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                  start_factor=0.0001,
                                                  total_iters=10)
    
    logger.info(f"Loss Function: {str(loss)}")
    logger.info(f"Optimizer: {str(optimizer)}")
    logger.info(f"Scheduler: {str(scheduler)}")


    logger.info(f"Starting training with {args.num_epochs} epochs")

    if "siamese" in args.model_config.lower():
        train_loop(model, train_loader, val_loader, loss, optimizer,
                args.num_epochs, device, writer, path_save,
                scheduler)
    else:
        train_loop_one_image(model, train_loader, val_loader, loss, optimizer,
                args.num_epochs, device, writer, path_save,
                scheduler)
    logger.info("Finishing training")

    torch.save(model.state_dict(), os.path.join(path_save, "final_model.pt"))
    
    writer.close()

               
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Necessary information to run the training pipeline")
    
    parser.add_argument("-num_classes", "--num_classes", default=2, type = int, help="Number of classes (default=2)")
    parser.add_argument("-modelconfig", "--model_config", required=True, help="Model \
                        selection: Resnet50, WRN50, ViT_B, ViT_S, Swin_B, Swin_S, SwinPr")
    parser.add_argument("-json_fold", "--json_fold", default="", help="JSON with train, val, test split")
    parser.add_argument("-b", "--batch_size", default=8, type = int, help="Batch Size used for training (Default=8)")
    parser.add_argument("-lr", "--learning_rate", default=2e-5, type = float, help="Learning Rate for training (Default=2e-5)")
    parser.add_argument("-epoch", "--num_epochs", default=50, type = int, help="Number of epochs to train the model (Default=50)")
    parser.add_argument("-img_h", "--img_harmonization", default="", help="Train using harmonized images (default is no)")
    parser.add_argument("-sampler", "--sampler", default=None, help="Balanced Sampler")
    parser.add_argument("-data_setup", "--data_setup", default="", help="Combination of images for training (Mix, Og+tf, None)")
    parser.add_argument("-log_directory", "--logs_dir", default = "logs")
    parser.add_argument("--save_dir", default = 'finetunedmodels')
    parser.add_argument("--model_naming", default = 'newmodel')
    parser.add_argument("-seed", "--seed", default=3108, type = int, help="Random seed")
    parser.add_argument("-layer", "--layer", default="", help="layer for linear probe")

    
    args_v = parser.parse_args()
    main(args_v)
