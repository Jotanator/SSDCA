# from swin_model import SwinTransformerClassification
from load_dataset import create_dataloader, create_dataloader_one_image
from tqdm import tqdm
import torch
import numpy as np
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score, roc_curve, auc, precision_recall_curve, PrecisionRecallDisplay, ConfusionMatrixDisplay
import argparse
from utils import get_model
from train import test_model_one_image, test_model_gradCam
import os
import openpyxl
np.set_printoptions(threshold=np.inf)  # disables truncation
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main(args):
    print("Starting Testing Pipeline")

    print(f"Using JSON: {args.json_fold}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device used: {device}")
    print("Start loading data")

    if "siamese" in args.model_config.lower():
        print("SIAMESE FOUND")
        train_loader, val_loader, test_loader = create_dataloader(args.batch_size,
                                                                args.json_fold,
                                                                args.sampler,
                                                                args.data_setup)
    else:
        train_loader, val_loader, test_loader = create_dataloader_one_image(args.batch_size,
                                                            args.json_fold,
                                                            args.sampler,
                                                            args.data_setup)                                                            

    print('Finished loading data')

    print(f"Starting model creation with model: {args.model_config}")
    model = get_model(args.model_config, args.layer, 2)
    root = args.load_folder
    print(root + "model_epoch" + args.load_epoch + ".pt")
   
    # Choose best Epoch: 
    path = glob.glob(root + "events.out.tfevents.*")[0]
    ea = EventAccumulator(path, size_guidance={'scalars': 0})
    ea.Reload()
    scalar_tags = set(ea.Tags().get('scalars', []))
    events = ea.Scalars("Val/Acc")
    events = [[e.step, float(e.value)] for e in events]
    max_val = 0
    chosen_epoch = 30
    for i in events:
        if max_val < i[1]:
            max_val = i[1]
            chosen_epoch = i[0]

    print("Best Epoch from Validation is: ", chosen_epoch)
    ckpt = torch.load(root + "model_epoch" + str(chosen_epoch) + ".pt", map_location='cpu')
    
    sd_fixed = {}
    for k, v in ckpt.items():
        nk = k
        while nk.startswith("module."):
            nk = nk[len("module."):]
        sd_fixed[nk] = v

    model.load_state_dict(sd_fixed, strict=True) 
    model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
        
    correct_predictions = 0
    model.eval()

    all_pred = []
    all_labels = []
    all_pred_prob = []
    all_paths1 = []
    all_paths2 = []
    correct_predictions = []
    total_labels = []
    all_names1 = []
    all_names2 = []
    embeddings = []

    if "siamese" in args.model_config.lower():
        all_pred, all_labels, all_pred_prob, all_paths1, all_paths2, correct_predictions, total_labels, all_names1, all_names2, embeddings = test_model_gradCam(model, test_loader, device)
    else:
        all_pred, all_labels, all_pred_prob, all_paths1, correct_predictions, total_labels, all_names1, embeddings = test_model_one_image(model, test_loader, device)

    
    all_pred = torch.cat(all_pred).detach().to('cpu')
    all_labels = torch.cat(all_labels).detach().to('cpu')
    all_pred_prob = torch.cat(all_pred_prob).detach().to('cpu')
    embeddings = torch.cat(embeddings).detach().to('cpu')
    assert(all_labels.shape[0] == all_pred.shape[0] == all_pred_prob.shape[0] == embeddings.shape[0])


    assert(all_labels.shape[0] == all_pred.shape[0] == all_pred_prob.shape[0])

    balanced_acc = balanced_accuracy_score(all_pred, all_labels)

    cm = confusion_matrix(all_labels, all_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(root + "conf_mat.png")
    
    tn, fp, fn, tp  = cm.ravel()

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    final_accuracy = 100. * correct_predictions / total_labels

    print("Total Sample", total_labels)
    print("Recall: ", recall)
    print("Precision: ", precision)
    print("Balanced Accuracy: ", balanced_acc)
    print("Non-Balanced Accuracy: ", final_accuracy)

    calc_roc_curve(all_pred_prob, all_labels, root)
    print(embeddings.shape)
    flat_embeddings = [e.cpu().numpy() for e in embeddings]  # list of (512,) arrays
    print(len(flat_embeddings))
    df = pd.DataFrame({
        'labels': all_labels.tolist(),
        'predictions': all_pred.tolist(),
        'probabilities': all_pred_prob.tolist(),
        'path1': all_paths1,
        'path2': all_paths2 if all_paths2 else [None] * len(all_labels),
        'name1': all_names1,
        'name2': all_names2 if all_names2 else [None] * len(all_labels),
        'Embeddings': flat_embeddings
    })
    np.savez(root+args.json_fold[:-5]+"_ExtraCases_Expanded_Epoch30_embeddings_newCropping_forAll_embeddings_IncludeVal.npz", Embeddings=flat_embeddings)

    
    # Save to Excel
    df.to_excel(root + args.json_fold[:-5] + "_ExtraCases_Expanded_Epoch30_embeddings_newCropping_forAll_IncludeVal.xlsx", index=False)

    print(round(balanced_acc, 4), round(recall, 4), round(precision, 4))


def calc_roc_curve(all_pred_prob, all_labels, root):
    fpr, tpr, _ = roc_curve(all_labels.numpy(), all_pred_prob.numpy())
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.savefig(root + "roc_curve.png")

    prec, recall, _ = precision_recall_curve(all_labels.numpy(), all_pred_prob.numpy())
    pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()
    plt.savefig(root + "prc_curve.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Necessary information to run the training pipeline")

    parser.add_argument("-modelconfig", "--model_config", required=True, help="Model \
                        selection: Resnet50, WRN50, ViT_B, ViT_S, Swin_B, Swin_S, SwinPr")

    parser.add_argument("-json_fold", "--json_fold", default="", help="JSON with train, val, test split")
    parser.add_argument("-load_folder", "--load_folder", default="", help="Name of folder path with model")
    parser.add_argument("-load_epoch", "--load_epoch", default="50", help="Epoch to load")

    parser.add_argument("-b", "--batch_size", default=1, type = int, help="Batch Size used for training (Default=8)")
    parser.add_argument("-epoch", "--num_epochs", default=50, type = int, help="Number of epochs to train the model (Default=50)")
    
    parser.add_argument("-img_h", "--img_harmonization", default="", help="Train using harmonized images (default is no)")
    parser.add_argument("-sampler", "--sampler", default=None, help="Balanced Sampler")
    parser.add_argument("-data_setup", "--data_setup", default="", help="Combination of images for training (Mix, Og+tf, None)")
    parser.add_argument("-layer", "--layer", default="", help="JSON with train, val, test split")

    args_v = parser.parse_args()
    main(args_v)


