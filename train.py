import os
import torch
from tqdm import tqdm
from gradCams import save_gradcam_overlay
import pandas as pd
from models.Swin_explainability import compute_final_saliency, compute_relation_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import islice

def train_loop_one_image(model, train_loader, val_loader, loss_f, optimizer, epochs, device, writer, path_save, scheduler):

    model.train()
    for epoch in range(epochs):
        avg_loss = 0
        correct_predictions = 0
        total_labels = 0
        train_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True) 

        for i, (img1, label1, path1) in enumerate(train_tqdm):
            img1, label1 = img1.to(device), label1.to(device)
            labels = label1
            optimizer.zero_grad()
            outputs = model(img1)
            loss = loss_f(outputs, label1)
            loss.backward()
            optimizer.step()

            # Save values
            avg_loss += loss.item()
            _, predicted = outputs.max(1)
            total_labels += labels.size(0)
            correct_predictions += predicted.eq(labels).sum().item()
            curr_loss = loss.item()
            curr_acc =  100. * correct_predictions/total_labels
            
            global_step = epoch * len(train_loader) + i
            writer.add_scalar("Train/Iteration_Loss", curr_loss, global_step)
            writer.add_scalar("Train/Iteration_Acc", curr_acc, global_step)
            writer.add_scalar("Train/LR", optimizer.param_groups[0]['lr'], global_step)

            train_tqdm.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{curr_acc:.2f}'
            })
        

        final_loss = avg_loss/len(train_loader)
        final_accuracy = 100. * correct_predictions / total_labels

        writer.add_scalar("Train/Epoch_Loss", final_loss, epoch+1)
        writer.add_scalar("Train/Epoch_Acc", final_accuracy, epoch+1)
        
        if (epoch+1) % 3 == 0:
            eval_model_one_image(model, val_loader, loss_f, device, epoch, writer)
            torch.save(model.state_dict(), os.path.join(path_save, 'model_epoch{}.pt'.format(epoch+1)))
            
        if scheduler is not None:
            scheduler.step()
            
    writer.flush()

def eval_model_one_image(model, val_loader, loss_f, device, epoch, writer):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_labels = 0
    val_tqdm = tqdm(val_loader, desc=f"Validating Model at Epoch {epoch+1}", leave=False) 
    with torch.no_grad():
        for i, (img1, label1, path1) in enumerate(val_tqdm):
            img1, label1 = img1.to(device), label1.to(device)
            labels = label1
            
            outputs = model(img1)
            loss = loss_f(outputs, label1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_labels += batch_size
            _, predicted = outputs.max(1)
            correct_predictions += predicted.eq(labels).sum().item()

            curr_acc = 100. * correct_predictions / total_labels
            val_tqdm.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{curr_acc:.2f}'
            })

    final_loss = total_loss / total_labels  # Corrected loss calculation
    final_accuracy = 100. * correct_predictions / total_labels

    # Log validation metrics
    writer.add_scalar("Val/Loss", final_loss, epoch+1)
    writer.add_scalar("Val/Acc", final_accuracy, epoch+1)


def train_loop(model, train_loader, val_loader, loss_f, optimizer, epochs, device, writer, path_save, scheduler):
    model.train()
    for epoch in range(epochs):
        avg_loss = 0
        correct_predictions = 0
        total_labels = 0
        train_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True) 
        for i, (img1, img2, label1, label2, path1, path2) in enumerate(train_tqdm):
            img1, img2, label1, label2 = img1.to(device), img2.to(device), label1.to(device), label2.to(device)
            
            assert list(label1) == list(label2)
            labels = label1

            optimizer.zero_grad()
            outputs, _ = model(img1, img2)
            loss = loss_f(outputs, labels)
            loss.backward()
            optimizer.step()

            # Save values
            avg_loss += loss.item()
            _, predicted = outputs.max(1)
            total_labels += labels.size(0)
            correct_predictions += predicted.eq(labels).sum().item()
            curr_loss = loss.item()
            curr_acc =  100. * correct_predictions/total_labels
            global_step = epoch * len(train_loader) + i
            writer.add_scalar("Train/Iteration_Loss", curr_loss, global_step)
            writer.add_scalar("Train/Iteration_Acc", curr_acc, global_step)
            writer.add_scalar("Train/LR", optimizer.param_groups[0]['lr'], global_step)

            train_tqdm.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{curr_acc:.2f}'
            })
        

        final_loss = avg_loss/len(train_loader)
        final_accuracy = 100. * correct_predictions / total_labels

        writer.add_scalar("Train/Epoch_Loss", final_loss, epoch+1)
        writer.add_scalar("Train/Epoch_Acc", final_accuracy, epoch+1)
        
        if (epoch+1) % 3 == 0:
            eval_model(model, val_loader, loss_f, device, epoch, writer)
            torch.save(model.state_dict(), os.path.join(path_save, 'model_epoch{}.pt'.format(epoch+1)))
            
        if scheduler is not None:
            scheduler.step()
            
    writer.flush()

def eval_model(model, val_loader, loss_f, device, epoch, writer):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_labels = 0
    val_tqdm = tqdm(val_loader, desc=f"Validating Model at Epoch {epoch+1}", leave=False) 
    with torch.no_grad():
        for i, (img1, img2, label1, label2, path1, path2) in enumerate(val_tqdm):
            img1, img2, label1, label2 = img1.to(device), img2.to(device), label1.to(device), label2.to(device)

            assert list(label1) == list(label2)
            assert os.path.basename(path1[0]).split('_')[0] == os.path.basename(path2[0]).split('_')[0]

            labels = label1

            outputs, _ = model(img1, img2)
            loss = loss_f(outputs, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_labels += batch_size
            _, predicted = outputs.max(1)

            correct_predictions += predicted.eq(labels).sum().item()

            curr_acc = 100. * correct_predictions / total_labels
            val_tqdm.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{curr_acc:.2f}'
            })

    final_loss = total_loss / total_labels  # Corrected loss calculation
    final_accuracy = 100. * correct_predictions / total_labels

    # Log validation metrics
    writer.add_scalar("Val/Loss", final_loss, epoch+1)
    writer.add_scalar("Val/Acc", final_accuracy, epoch+1)


def test_model_one_image(model, test_loader, device):
    model.eval()
    all_pred = []
    all_labels = []
    all_pred_prob = []
    all_paths1 = []
    all_names1 = []
    embeddings = []
    correct_predictions = 0
    total_labels = 0
    test_tqdm = tqdm(test_loader, desc=f"Testing Model:", leave=False) 
    with torch.no_grad():
        for i, (img1, label1, path1) in enumerate(test_tqdm):
            img1, label1 = img1.to(device), label1.to(device)

            labels = label1

            final_predictions = torch.tensor([]).to(device)
            outputs_logits = model(img1)
            outputs = torch.softmax(outputs_logits, dim=1)
            embedding = model.forward_features(img1)
            # embedding = model.global_pool(embedding)
            embedding = model.forward_head(embedding, pre_logits=True)

            # Save values
            probs, predicted = outputs.max(1)

            total_labels += labels.size(0)
            correct_predictions += predicted.eq(labels).sum().item()

            all_pred_prob.append(outputs[:,1])
            all_pred.append(predicted)
            all_labels.append(labels)
            all_paths1.extend(path1)
            embeddings.append(embedding)

            names1 = [os.path.basename(p) for p in path1]
            all_names1.extend(names1)
            curr_acc = 100. * correct_predictions / total_labels
            test_tqdm.set_postfix({
                'Acc': f'{curr_acc:.2f}'
            })

    return all_pred, all_labels, all_pred_prob, all_paths1, correct_predictions, total_labels, all_names1, embeddings

import torch.nn.functional as F

def plot_attention_from_embeddings(img0, img1, attn_emb2, attn_emb1, prediction, label2, name):
    """
    image: [3, H, W], original image
    attn_emb: [7, 7, 768], attention output embeddings
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # De-normalize image
    img = img0.clone().cpu()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.clamp(0, 1)
    img_np = img.permute(1, 2, 0).numpy()


    img2 = img1.clone().cpu()
    for c in range(3):
        img2[c] = img2[c] * std[c] + mean[c]
    img2 = img2.clamp(0, 1)
    img2_np = img2.permute(1, 2, 0).numpy()

    save_path='gradCams/AttentionMaps_modified'
    # Reduce channel dim → [7, 7]
    attn_map1 = attn_emb1.mean(dim=-1)  # average across 768 channels
    attn_map2 = attn_emb2.mean(dim=-1)  # average across 768 channels
    
    # Normalize to [0, 1]
    attn_map1 = (attn_map1 - attn_map1.min()) / (attn_map1.max() - attn_map1.min())

    # Upsample to image size
    attn_map_t = attn_map1.unsqueeze(0).unsqueeze(0)  # [1,1,7,7]
    attn_map1 = F.interpolate(attn_map_t, size=img1.shape[1:], mode='bilinear', align_corners=False)
    attn_map1 = attn_map1.squeeze().cpu().numpy()

        
    # Normalize to [0, 1]
    attn_map2 = (attn_map2 - attn_map2.min()) / (attn_map2.max() - attn_map2.min())

    # Upsample to image size
    attn_map2 = attn_map2.unsqueeze(0).unsqueeze(0)  # [1,1,7,7]
    attn_map2 = F.interpolate(attn_map2, size=img1.shape[1:], mode='bilinear', align_corners=False)
    attn_map2 = attn_map2.squeeze().cpu().numpy()

    # Plot
    fig, ax = plt.subplots(1, 4, figsize=(10, 6))
    ax[0].imshow(img_np)
    ax[0].set_title("Original Image 1 - Restaging")
    ax[0].axis("off")

    ax[1].imshow(img_np)
    ax[1].imshow(attn_map1, cmap='jet', alpha=0.5)
    ax[1].set_title("Attention Map")
    ax[1].axis("off")

    ax[2].imshow(img2_np)
    ax[2].set_title("Original Image Follow up")
    ax[2].axis("off")

    ax[3].imshow(img2_np)
    ax[3].imshow(attn_map2, cmap='jet', alpha=0.5)
    ax[3].set_title("Attention Map")
    ax[3].axis("off")



    filename = "Img_2" + "_Pred_" + str(prediction) + "_" + "label_" + str(label2) + "_" + name
    plt.savefig(os.path.join(save_path, filename), format='jpg', dpi=300)
    plt.close()

def idx_to_rc(idx, Wp):
    """Row-major index -> (row, col) in patch grid."""
    r = idx // Wp
    c = idx % Wp
    return int(r), int(c)

def patch_xywh(r, c, H, W, Hp, Wp):
    """
    Pixel coords (x,y,width,height) for patch (r,c) in an HxW image
    given a Hp x Wp patch grid.
    """
    ph = H / Hp
    pw = W / Wp
    x = c * pw
    y = r * ph
    return x, y, pw, ph

def draw_patch_bbox(ax, r, c, H, W, Hp, Wp, color="lime", lw=1, label=None):
    """Draw a rectangle around patch (r,c)."""
    x, y, pw, ph = patch_xywh(r, c, H, W, Hp, Wp)
    rect = mpatches.Rectangle((x, y), pw, ph, fill=False, ec=color, lw=lw)
    ax.add_patch(rect)
    # if label is not None:
    #     ax.text(x + pw/2, y + ph/2, str(label),
    #             ha="center", va="center", color=color, fontsize=9, weight="bold")
        
def plot_attention(img0, img1, attn_emb2, attn_emb1, prediction, label2, name):
    """
    image: [3, H, W], original image
    attn_emb: [7, 7, 768], attention output embeddings
    """

    for i in range(attn_emb1.shape[0]):
        attn_map1 = attn_emb1[i].cpu().view(7, 7).detach()
        attn_map2 = attn_emb2[i].cpu().view(7, 7).detach()

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        # De-normalize image
        img = img0.clone().cpu()
        for c in range(3):
            img[c] = img[c] * std[c] + mean[c]
        img = img.clamp(0, 1)
        img_np = img.permute(1, 2, 0).numpy()


        img2 = img1.clone().cpu()
        for c in range(3):
            img2[c] = img2[c] * std[c] + mean[c]
        img2 = img2.clamp(0, 1)
        img2_np = img2.permute(1, 2, 0).numpy()

        # print("save image")
        # save_path=f'gradCams/AttentionMaps_modified/{name[:-4]}'
        # if not os.path.exists(save_path):
        #     os.mkdir(save_path)

        base_path = f'gradCams/AttentionMaps_modified/{name[:-4]}'
        patch_path = os.path.join(base_path, f'Patch_{i}')
        os.makedirs(patch_path, exist_ok=True)


        attn_map1 = attn_map1.unsqueeze(0).unsqueeze(0)          # (1,1,7,7)
        attn_map1 = F.interpolate(attn_map1, size=(224,224), mode="bilinear", align_corners=False)
        attn_map1 = attn_map1[0,0].cpu().numpy()  # (224,224)

        attn_map2 = attn_map2.unsqueeze(0).unsqueeze(0)          # (1,1,7,7)
        attn_map2 = F.interpolate(attn_map2, size=(224,224), mode="bilinear", align_corners=False)
        attn_map2 = attn_map2[0,0].cpu().numpy()  # (224,224)


        # 1. Original Image 1 - Restaging with bbox
        r, c = idx_to_rc(i, 7)
        fig, ax = plt.subplots()
        ax.imshow(img_np)
        draw_patch_bbox(ax, r, c, 224, 224, 7, 7, color="lime", lw=1)
        ax.axis("off")
        fig.savefig(os.path.join(patch_path, f"restaging_{prediction}_{label2}_{name}.jpg"),
                    format='jpg', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # 2. Image 2 with attn_map1 overlay
        fig, ax = plt.subplots()
        ax.imshow(img2_np)
        ax.imshow(attn_map1, alpha=0.6)
        ax.axis("off")
        fig.savefig(os.path.join(patch_path, f"attn1_{prediction}_{label2}_{name}.jpg"),
                    format='jpg', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # 3. Original Image 2 - Follow up with bbox
        fig, ax = plt.subplots()
        ax.imshow(img2_np)
        draw_patch_bbox(ax, r, c, 224, 224, 7, 7, color="lime", lw=1)
        ax.axis("off")
        fig.savefig(os.path.join(patch_path, f"followup_{prediction}_{label2}_{name}.jpg"),
                    format='jpg', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # 4. Image 1 with attn_map2 overlay
        fig, ax = plt.subplots()
        ax.imshow(img_np)
        ax.imshow(attn_map2, alpha=0.6)
        ax.axis("off")
        fig.savefig(os.path.join(patch_path, f"attn2_{prediction}_{label2}_{name}.jpg"),
                    format='jpg', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        # Plot
        # fig, ax = plt.subplots(1, 4, figsize=(10, 6))


        # ax[0].imshow(img_np)
        # r, c = idx_to_rc(i, 7)
        # draw_patch_bbox(ax[0], r, c, 224, 224, 7, 7, color="lime", lw=1, label=(i if True else None))
        # ax[0].set_title("Original Image 1 - Restaging")
        # ax[0].axis("off")

        # ax[1].imshow(img2_np)
        # ax[1].imshow(attn_map1, alpha=0.6)
        # ax[1].set_title("Attention Map")
        # ax[1].axis("off")

        # ax[2].imshow(img2_np)
        # draw_patch_bbox(ax[2], r, c, 224, 224, 7, 7, color="lime", lw=1, label=(i if True else None))
        # ax[2].set_title("Original Image Follow up")
        # ax[2].axis("off")

        # ax[3].imshow(img_np)
        # ax[3].imshow(attn_map2, alpha=0.6)
        # ax[3].set_title("Attention Map")
        # ax[3].axis("off")

        # filename = "Patch_" + str(i) + "Img_2" + "_Pred_" + str(prediction) + "_" + "label_" + str(label2) + "_" + name
        # plt.savefig(os.path.join(save_path, filename), format='jpg', dpi=300)
        # plt.close()

def rotate_rgb(img, deg):
    return torch.rot90(img, (int(deg)//90)%4, dims=(-2,-1)).contiguous()

import torch.nn.functional as F
def zoom_rgb(img, z):
    B,C,H,W = img.shape; assert B==1
    z = max(float(z), 1.0)
    h, w = max(1, int(H/z)), max(1, int(W/z))
    y, x = (H-h)//2, (W-w)//2
    return F.interpolate(img[:,:,y:y+h, x:x+w], (H,W), mode='bilinear', align_corners=False)

def test_model_gradCam(model, test_loader, device):
    model.eval()
    all_pred = []
    all_labels = []
    all_pred_prob = []
    all_paths1 = []
    all_paths2 = []
    all_names1 = []
    all_names2 = []
    embeddings = []
    correct_predictions = 0
    total_labels = 0
    df = pd.read_excel("./DistanceFromLR/SwinSmall_fold_2_output_epoch30_115.xlsx")
    df['basename'] = df['path1'].apply(os.path.basename)
    test_tqdm = tqdm(test_loader, desc=f"Testing Model:", leave=False) 
    for i, (img1, img2, label1, label2, path1, path2) in enumerate(test_tqdm):
        # if path2[0] != "/lab/deasylab1/Jorge/Endo_2025_Temporal/datasets/classification/MSK/LR/Test/Tumor/5feca24a_10_22_2021_post8c_LR.jpg":
        #     continue
        img1, img2, label1, label2 = img1.to(device), img2.to(device), label1.to(device), label2.to(device)
        assert list(label1) == list(label2)
        assert os.path.basename(path1[0]).split('_')[0] == os.path.basename(path2[0]).split('_')[0]

        with torch.no_grad():
            img1, img2, label1, label2 = img1.to(device), img2.to(device), label1.to(device), label2.to(device)
            assert list(label1) == list(label2)
            assert os.path.basename(path1[0]).split('_')[0] == os.path.basename(path2[0]).split('_')[0]
            labels = label1
            final_predictions = torch.tensor([]).to(device)
            # img2 = rotate_rgb(img2, 270)
            # img2 = zoom_rgb(img2, 2.0)
            # outputs_logits, embedding, attn_embed2, attn_embed1 = model(img1, img2)
            outputs_logits, embedding = model(img1, img2)
            outputs = torch.softmax(outputs_logits, dim=1)
            # Save values
            probs, predicted = outputs.max(1)
            # if os.path.basename(path1[0]) in df['basename'].tolist():
            #     plot_attention_from_embeddings(img1[0], img2[0], attn_embed2[0], attn_embed1[0], predicted[0].item(), label2[0].item(), os.path.basename(path2[0]))
            # print("Generate attention map")
            # if os.path.basename(path1[0]) in df['basename'].tolist():
            #     print("Plotting attention map")
            #     plot_attention(img1[0], img2[0], attn_embed2[0], attn_embed1[0], predicted[0].item(), label2[0].item(), os.path.basename(path2[0]))

            total_labels += labels.size(0)
            correct_predictions += predicted.eq(labels).sum().item()

            all_pred_prob.append(outputs[:,1])
            all_pred.append(predicted)
            all_labels.append(labels)
            all_paths1.extend(path1)
            all_paths2.extend(path2)
            names1 = [os.path.basename(p) for p in path1]
            names2 = [os.path.basename(p) for p in path2]
            all_names1.extend(names1)
            all_names2.extend(names2)
            embeddings.append(embedding)
            # print(embeddings)
            # exit()
            curr_acc = 100. * correct_predictions / total_labels
            test_tqdm.set_postfix({
                'Acc': f'{curr_acc:.2f}'
            })
        # if os.path.basename(path1[0]) not in df['basename'].tolist():
        #         continue
        
        # name_img =""

        # # 1) choose the class logit to backprop from (use predicted)
        # # img = img2
        # name_img = os.path.basename(path2[0])
        # output, _ = model(img1, img2)

        # probs, predicted = output.max(1)
        # score = output[0, predicted[0]]
        # model.zero_grad()
        # score.backward(retain_graph=True)

        # # 2) fetch hook tensors
        # acts  = model.activations['feat']
        # grads = model.gradients['feat']

        # acts_map  = acts.permute(0, 3, 1, 2).contiguous()
        # grads_map = grads.permute(0, 3, 1, 2).contiguous()

        # # 5) Grad-CAM weights = global-average of gradients over spatial dims
        # weights = grads_map.mean(dim=(2, 3), keepdim=True)                   # [B, C, 1, 1]

        # # 6) weighted sum over channels -> CAM, ReLU, normalize
        # cam = (weights * acts_map).sum(dim=1, keepdim=True)                  # [B, 1, H, W]
        # cam = F.relu(cam)

        # # normalize per-sample
        # eps = 1e-8
        # cam = cam - cam.amin(dim=(2,3), keepdim=True)
        # cam = cam / (cam.amax(dim=(2,3), keepdim=True) + eps)

        # # 7) upsample CAM to input image size (assumes img is [B,3,Himg,Wimg])
        # cam_up = F.interpolate(cam, size=img2.shape[-2:], mode='bilinear', align_corners=False)

        # # 8) your overlay/save (keep your function)
        # save_gradcam_overlay(
        #     input_tensor=img1[0],             # or img1[0] for the other view
        #     cam=cam_up[0].unsqueeze(0),      # [1,1,Himg,Wimg]
        #     image_2=img2[0],
        #     prediction=predicted[0].item(),
        #     label=label1[0].item(),
        #     save_path='gradCams/gradcams/',
        #     index=1,
        #     combination_path=f"combination_{i}",
        #     name=name_img
        # )

        # img = img2
        # name_img = os.path.basename(path2[0])
        # output, _ = model(img1, img2)
        # output[0][0].backward(retain_graph=True)
        # probs, predicted = output.max(1)
        # # # Grad-CAM from shared backbone layer
        # acts = model.activations['feat']
        # grads = model.gradients['feat']
        # weights = grads.mean(dim=(2, 3), keepdim=True)
        # cam = (weights * acts).sum(dim=1, keepdim=True)
        # cam = torch.relu(cam)
        # save_gradcam_overlay(
        #     input_tensor=img[0],       # or img1[0] if you want that view
        #     cam=cam[0].unsqueeze(0),
        #     image_2=img1[0],
        #     prediction=predicted[0].item(),
        #     label = label1[0].item(),
        #     save_path='gradCams/gradcams/',
        #     index=1,
        #     combination_path= "combination_" + str(i),
        #     name = name_img
        # )
        # torch.cuda.empty_cache()
        # torch.cuda.ipc_collect()
        # del output, acts, grads, cam


    return all_pred, all_labels, all_pred_prob, all_paths1, all_paths2, correct_predictions, total_labels, all_names1, all_names2, embeddings
