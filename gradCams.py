import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

def save_gradcam_overlay(
    input_tensor,
    image_2,
    cam,
    prediction,
    label,
    save_path,
    combination_path,
    index=0,
    name="",
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
):
    
    save_path = save_path
        
    # De-normalize image
    img = input_tensor.clone().cpu()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.clamp(0, 1)
    img_np = img.permute(1, 2, 0).numpy()


    img2 = image_2.clone().cpu()
    for c in range(3):
        img2[c] = img2[c] * std[c] + mean[c]
    img2 = img2.clamp(0, 1)
    img2_np = img2.permute(1, 2, 0).numpy()

    # Normalize and upsample CAM
    if cam.ndim == 3:
        cam = cam.unsqueeze(0)  # (1, C, H, W)
    elif cam.ndim == 2:
        cam = cam.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    cam = F.interpolate(cam, size=img_np.shape[:2], mode='bilinear', align_corners=False).squeeze().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-5)

    # Convert CAM to 3-channel mask
    # cam_rgb = np.stack([cam]*3, axis=-1)  # shape: (H, W, 3)

    # Blend image with attention mask (dark areas = low attention)
    # blended = img_np * cam_rgb

    base_path = f'gradCams/gradcams/{name[:-4]}'
    os.makedirs(base_path, exist_ok=True)

    # Plot and save
    fig, ax = plt.subplots()
    ax.imshow(img2_np)
    # ax.set_title("Restaging Image")
    ax.axis('off')
    filename = "img2_original" + name + ".jpg"
    fig.savefig(os.path.join(base_path, filename),
                format='jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.imshow(img_np)
    # ax.set_title("Follow up or LR Image")
    ax.axis('off')
    filename = "img1_original" + name + ".jpg"
    fig.savefig(os.path.join(base_path, filename),
                format='jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    fig, ax = plt.subplots()

    ax.imshow(img_np)
    ax.imshow(cam, cmap='jet', alpha=0.4)
    ax.axis('off')
    filename = "img1_GradCam_" + name + ".jpg"
    plt.savefig(os.path.join(base_path, filename), format='jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    fig, ax = plt.subplots()
    ax.imshow(img2_np)
    ax.imshow(cam, cmap='jet', alpha=0.4)
    ax.axis('off')
    filename = "img2_GradCam_" + name + ".jpg"
    plt.savefig(os.path.join(base_path, filename), format='jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
