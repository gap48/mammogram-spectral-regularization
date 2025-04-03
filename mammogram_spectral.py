import os
import sys
import glob
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import logging
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
from scipy import ndimage
from skimage import filters, morphology, measure
from torch.utils.data import Dataset, DataLoader, random_split, Subset

import monai
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    RandFlipd,
    RandZoomd,
    RandGaussianNoised,
    Resized,
    ToTensord,
    MapTransform,
    Compose
)
import timm

from tqdm import tqdm
from sklearn.metrics import classification_report
from typing import Optional, Tuple, Union


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#########################################################
# 1) DATASET CLASSES (Unchanged)
#########################################################

class ComputeMask(MapTransform):
    def __init__(self, keys, threshold=0.01):
        super().__init__(keys)
        self.threshold = threshold

    def segment_breast(self, image):
        image = image.astype(np.float32)
        if image.max() > 0:
            image = image / max(1e-8, image.max())
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply((image * 255).astype(np.uint8))
        enhanced = enhanced.astype(np.float32) / 255.0
        thresh_1 = filters.threshold_otsu(enhanced)
        binary_1 = enhanced > thresh_1
        dark_regions = enhanced[enhanced <= thresh_1]
        if len(dark_regions) > 0:
            thresh_2 = filters.threshold_otsu(dark_regions)
            binary_2 = enhanced > thresh_2
            binary = np.logical_or(binary_1, binary_2)
        else:
            binary = binary_1
        binary = morphology.remove_small_objects(binary, min_size=1000)
        binary = morphology.remove_small_holes(binary, area_threshold=1000)
        labels = measure.label(binary)
        props = measure.regionprops(labels)
        if props:
            largest_region = max(props, key=lambda p: p.area)
            binary = (labels == largest_region.label)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        binary = cv2.dilate(binary.astype(np.uint8), kernel, iterations=2)
        binary = cv2.erode(binary.astype(np.uint8), kernel, iterations=1)
        binary = ndimage.binary_fill_holes(binary).astype(np.float32)
        return binary

    def __call__(self, data):
        for key in self.keys:
            img = data[key].clone()
            if img.ndim == 3 and img.shape[0] == 1:
                mask_1ch = self.segment_breast(img[0].clone())
                data["mask"] = torch.tensor(mask_1ch, dtype=torch.float32).unsqueeze(0)
            else:
                raise ValueError("Expected single-channel shape [1, H, W].")
        return data

class MammoReconstructionDataset(Dataset):
    def __init__(self, data_source, transform=None):

        if hasattr(data_source, 'iloc'):
        
            if isinstance(data_source.index, pd.RangeIndex):
         
                self.is_dataframe = True
                self.dataframe = data_source
                self.path_column = data_source.columns[0]
                self.file_paths = None  
            else:
                self.is_dataframe = False
                self.dataframe = None
                self.file_paths = data_source.iloc[:, 0].tolist()
        else:
            self.is_dataframe = False
            self.dataframe = None
            self.file_paths = list(data_source)

        self.transform = transform
        print(f"[INFO] Created MammoReconstructionDataset with {self.__len__()} images")
        print(f"[INFO] Dataset type: {'DataFrame-based' if self.is_dataframe else 'List-based'}")

    def __len__(self):
        if self.is_dataframe:
            return len(self.dataframe)
        else:
            return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            if self.is_dataframe:
                path = self.dataframe.iloc[idx][self.path_column]
            else:
                path = self.file_paths[idx]

            data = {'image': path}

            if self.transform:
                try:
                    loader = LoadImaged(keys='image', ensure_channel_first=True)
                    data = loader(data)

                    data = self.transform(data)

                    img = data['image']
                    if img.ndim != 3 or img.shape[0] != 1 or img.shape[1:] != (224, 224):
                        raise ValueError(f"Unexpected image shape: {img.shape}")
                except Exception as e:
                    logging.error(f"Error processing {path}: {str(e)}")
                    raise
            else:
                loader = LoadImaged(keys='image', ensure_channel_first=True)
                data = loader(data)
                img = data['image']
                mask = (img > 0.01).float()
                data = {'image': img, 'mask': mask}

            return data

        except Exception as e:
            logging.error(f"Error accessing index {idx}: {str(e)}")
            raise IndexError(f"Dataset access error at index {idx}: {str(e)}")

class MammoMetadataDataset(Dataset):
    def __init__(self, metadata_df, transform=None):
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        image_path = row['image_path']
        label_dict = {
            'mass_calc': row['label'],
            'pathology': row['pathology'],
            'subtlety': row['subtlety'],
            'breast_density': row['breast_density'],
            'assessment': row['assessment'],
            'abnormality_type': row['abnormality_type']
        }
        data = {'image': image_path}
        if self.transform:
            try:
                loader = LoadImaged(keys='image', ensure_channel_first=True)
                data = loader(data)
                data = self.transform(data)
                img = data['image']
                if img.ndim != 3 or img.shape[0] != 1 or img.shape[1:] != (224, 224):
                    raise ValueError(f"Unexpected image shape: {img.shape}")
            except Exception as e:
                logging.error(f"Error processing {image_path}: {str(e)}")
                raise
        else:
            loader = LoadImaged(keys='image', ensure_channel_first=True)
            data = loader(data)
            img = data['image']
            mask = (img > 0.01).float()
            data = {'image': img, 'mask': mask}
        data['labels'] = label_dict
        return data

def custom_collate(batch):
    images = torch.stack([item['image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    if 'labels' in batch[0]:
        labels = [item['labels'] for item in batch]
        collated_labels = {key: [d[key] for d in labels] for key in labels[0].keys()}
        return images, masks, collated_labels
    return images, masks

def plot_image_and_mask(image_tensor, mask_tensor=None, title="Step", output_dir=".", filename="step_debug.png", show_plot=False):
    if hasattr(image_tensor, 'detach'):
        image = image_tensor.detach().cpu().squeeze().numpy()
    else:
        image = image_tensor.squeeze()
    fig, axs = plt.subplots(1, 2 if mask_tensor is not None else 1, figsize=(10, 5))
    if mask_tensor is not None:
        if hasattr(mask_tensor, 'detach'):
            mask = mask_tensor.detach().cpu().squeeze().numpy()
        else:
            mask = mask_tensor.squeeze()
        axs = axs if isinstance(axs, np.ndarray) else [axs]
        axs[0].imshow(image, cmap='gray')
        axs[0].set_title(f"{title} - Image")
        axs[0].axis('off')
        axs[1].imshow(mask, cmap='gray')
        axs[1].set_title(f"{title} - Mask")
        axs[1].axis('off')
    else:
        axs.imshow(image, cmap='gray')
        axs.set_title(title)
        axs.axis('off')
    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=100)
    print(f"[DEBUG] Saved: {save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close()

def debug_transform_sequence(single_dcm_path, output_dir="."):
    data = {'image': single_dcm_path}
    loader = LoadImaged(keys='image', ensure_channel_first=True)
    data = loader(data)
    plot_image_and_mask(data['image'], None, title="Original Loaded Image", output_dir=output_dir, filename="debug_00_loaded.png")
    transform_sequence = [
        ComputeMask(keys=['image'], threshold=0.01),
        ScaleIntensityRanged(keys='image', a_min=0, a_max=65535, b_min=0.0, b_max=1.0, clip=True),
        RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0),
        RandZoomd(keys=['image', 'mask'], min_zoom=0.9, max_zoom=1.1, prob=0.3),
        RandGaussianNoised(keys='image', prob=0.2),
        Resized(keys=['image', 'mask'], spatial_size=(224, 224)),
        ToTensord(keys=['image', 'mask'])
    ]
    for idx, t in enumerate(transform_sequence, start=1):
        data = t(data)
        step_name = f"debug_{idx:02d}_{t.__class__.__name__}.png"
        plot_image_and_mask(data['image'], data.get('mask', None), title=f"After {t.__class__.__name__}", output_dir=output_dir, filename=step_name)
    print("[DEBUG] Completed step-by-step transform visualization.")

#########################################################
# 2) DATA LOADING
#########################################################

def find_full_mammogram(dcm_files, patient_folder=None):
    if patient_folder:
        return [dcm_path for dcm_path in dcm_files if patient_folder in dcm_path and "full mammogram images" in dcm_path]
    return [dcm_path for dcm_path in dcm_files if "full mammogram images" in dcm_path]

def create_mim_dataloaders(root_dir, dcm_files, batch_size=4, num_workers=4, seed=42, cache_dir='./cache', force_recompute=False):
    cache_file = os.path.join(cache_dir)

    if not force_recompute and os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            dcm_files = pickle.load(f)
        print(f"[INFO] Loaded {len(dcm_files)} MIM DICOM files from cache: {cache_file}")
    else:
        dcm_files = find_full_mammogram(dcm_files)
        with open(cache_file, 'wb') as f:
            pickle.dump(dcm_files, f)
        print(f"[INFO] Scanned and cached {len(dcm_files)} MIM DICOM files to {cache_file}")

    transform = Compose([
        ComputeMask(keys=['image'], threshold=0.01),
        ScaleIntensityRanged(keys='image', a_min=0, a_max=65535, b_min=0.0, b_max=1.0, clip=True),
        RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0),
        RandZoomd(keys=['image', 'mask'], min_zoom=0.9, max_zoom=1.1, prob=0.3),
        RandGaussianNoised(keys='image', prob=0.2),
        Resized(keys=['image', 'mask'], spatial_size=(224, 224)),
        ToTensord(keys=['image', 'mask'])
    ])
    full_dataset = MammoReconstructionDataset(dcm_files, transform=transform)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(seed))
    print(f"[MIM] Train set: {len(train_dataset)} | Val set: {len(val_dataset)} | Test set: {len(test_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=custom_collate)
    return train_loader, val_loader, test_loader

def create_partial_dataloaders(metadata_df, batch_size=4, num_workers=2, missing_label_percentage=100, seed=42):
    transform = Compose([
        ComputeMask(keys=['image'], threshold=0.01),
        ScaleIntensityRanged(keys='image', a_min=0, a_max=65535, b_min=0.0, b_max=1.0, clip=True),
        RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0),
        RandZoomd(keys=['image', 'mask'], min_zoom=0.9, max_zoom=1.1, prob=0.3),
        RandGaussianNoised(keys='image', prob=0.2),
        Resized(keys=['image', 'mask'], spatial_size=(224, 224)),
        ToTensord(keys=['image', 'mask'])
    ])
    dataset = MammoMetadataDataset(metadata_df, transform=transform)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(seed))
    if missing_label_percentage < 100:
        num_samples = int(missing_label_percentage / 100.0 * train_size)
        indices = torch.randperm(train_size)[:num_samples]
        train_dataset = Subset(train_dataset, indices)
        print(f"[CLS] Using {missing_label_percentage}% labeled training => {len(train_dataset)} samples.")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=custom_collate)
    return train_loader, val_loader, test_loader

#########################################################
# 4) UTILITY FUNCTIONS
#########################################################

def convert_pathology_to_label(pathology_list):
    numeric_labels = [1 if p.upper().startswith('MAL') else 0 for p in pathology_list]
    return torch.tensor(numeric_labels, dtype=torch.long)

#########################################################
# 5) MIM MODEL & TRAINING
#########################################################

class MaskedDecoder(nn.Module):
    def __init__(self, in_dim=1024, out_ch=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, 512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv_out = nn.Conv2d(64, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.deconv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.deconv2(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.deconv3(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.conv_out(x)
        return x

def breast_aware_random_mask(img, mask, mask_ratio=0.5, block_size=16, threshold=0.5):

    B, C, H, W = img.shape
    num_blocks_h = H // block_size
    num_blocks_w = W // block_size

    mask_avg = F.avg_pool2d(mask, kernel_size=block_size, stride=block_size)
    eligible_blocks = (mask_avg > threshold).view(B, -1)

    mask_tensor = torch.zeros((B, 1, H, W), dtype=torch.bool, device=img.device)
    for b in range(B):
        eligible_indices = torch.where(eligible_blocks[b])[0]
        num_eligible = eligible_indices.size(0)
        if num_eligible > 0:
            num_masked = int(mask_ratio * num_eligible)
            perm = torch.randperm(num_eligible, device=img.device)
            masked_indices = eligible_indices[perm[:num_masked]]
            for idx in masked_indices:
                row = idx // num_blocks_w
                col = idx % num_blocks_w
                row_start = row * block_size
                col_start = col * block_size
                mask_tensor[b, 0, row_start:row_start+block_size, col_start:col_start+block_size] = True

    masked_img = img.clone()
    masked_img[mask_tensor] = 0.0
    return masked_img, mask_tensor

def mask_guided_attention_regularization(attn_weights, breast_masks, block_size=16, lambda_reg=0.1):

    if len(attn_weights.shape) == 4:
        nW_B, num_heads, N, _ = attn_weights.shape

        attn_copy = attn_weights.detach().clone()

        attn_avg_per_head = []
        for h in range(num_heads):
            head_attn = attn_copy[:, h]  
            attn_avg_per_head.append(head_attn.mean())

        attn_avg = torch.stack(attn_avg_per_head).mean()

    elif len(attn_weights.shape) == 3:
        nW_B, N, _ = attn_weights.shape
        attn_copy = attn_weights.detach().clone()
        attn_avg = attn_copy.mean()
    else:
        return attn_weights, torch.tensor(0.0, device=attn_weights.device)

    B = breast_masks.shape[0]

    window_size = int(N ** 0.5) 

    mask_pooled = F.avg_pool2d(breast_masks, kernel_size=block_size, stride=block_size)
    mask_flat = mask_pooled.flatten(1).mean(dim=1)

    breast_ratio = mask_flat.mean()
    background_ratio = 1.0 - breast_ratio

    reg_term = lambda_reg * background_ratio * attn_avg

    return attn_weights, reg_term

@torch.no_grad()
def evaluate_mim(data_loader, encoder, decoder, criterion, mask_ratio=0.5, block_size=16):
    encoder.eval()
    decoder.eval()
    total_loss, total_samples = 0.0, 0
    for imgs, breast_masks in tqdm(data_loader, desc="Evaluating MIM", leave=False):
        imgs = imgs.cuda(non_blocking=True)
        breast_masks = breast_masks.cuda(non_blocking=True)
        masked_imgs, _ = breast_aware_random_mask(imgs, breast_masks, mask_ratio, block_size)
        feats = encoder.forward_features(masked_imgs)
        feats = feats.permute(0, 3, 1, 2)
        reconstructed = decoder(feats)
        reconstructed_upsampled = F.interpolate(reconstructed, size=(224, 224), mode='bilinear')
        loss = criterion(reconstructed_upsampled, imgs) * breast_masks
        loss = loss.sum() / (breast_masks.sum() + 1e-8)
        total_loss += loss.item() * imgs.size(0)
        total_samples += imgs.size(0)
    return total_loss / total_samples if total_samples > 0 else 0

def visualize_mim_batch(imgs, breast_masks, masked_imgs, epoch, output_dir='.', log_mode='console', batch_idx=0):
    img = imgs[0].detach().cpu().squeeze().numpy()
    bm = breast_masks[0].detach().cpu().squeeze().numpy()
    mm = masked_imgs[0].detach().cpu().squeeze().numpy()
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    axs[1].imshow(bm, cmap='gray')
    axs[1].set_title('Breast Mask')
    axs[1].axis('off')
    axs[2].imshow(mm, cmap='gray')
    axs[2].set_title('Masked Image')
    axs[2].axis('off')
    plt.suptitle(f"MIM Visualization - Epoch {epoch+1}, Batch {batch_idx}")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"mim_visual_epoch{epoch+1}_batch{batch_idx}.png")
    plt.savefig(plot_path)
    print(f"[INFO] MIM visual saved at: {plot_path}")
    if log_mode == "console":
        plt.show()
    else:
        plt.close()

def train_mim(encoder, decoder, train_loader, val_loader, test_loader, lr=1e-4, weight_decay=1e-5, num_epochs=5, output_dir='.', mask_ratio=0.5, block_size=16, log_mode='console'):
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    criterion = nn.L1Loss(reduction='none')
    optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=lr, weight_decay=weight_decay)
    best_test_loss = float('inf')
    train_losses, val_losses, test_losses = [], [], []

    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        train_loss, train_samples = 0.0, 0
        batch_pbar = tqdm(train_loader, desc=f"MIM Epoch {epoch+1}/{num_epochs}",
                          leave=False, file=sys.stdout, mininterval=0.1)
        for batch_idx, (imgs, breast_masks) in enumerate(batch_pbar):
            imgs = imgs.cuda(non_blocking=True)
            breast_masks = breast_masks.cuda(non_blocking=True)
            masked_imgs, _ = breast_aware_random_mask(imgs, breast_masks, mask_ratio, block_size)

            if batch_idx == 0:
                visualize_mim_batch(imgs, breast_masks, masked_imgs, epoch, output_dir, log_mode, batch_idx)

            handles = []
            reg_losses = []

            def get_hook(name):
                def hook(module, input, output):
                    _, reg_loss = mask_guided_attention_regularization(
                        output[0] if isinstance(output, tuple) else output,
                        breast_masks,
                        block_size=block_size,
                        lambda_reg=0.1
                    )

                    reg_losses.append(reg_loss)

                    return output

                return hook

            for name, module in encoder.named_modules():
                if "layers.3.blocks" in name and "attn" in name:
                    handles.append(module.register_forward_hook(get_hook(name)))

            feats = encoder.forward_features(masked_imgs)
            feats = feats.permute(0, 3, 1, 2)
            reconstructed = decoder(feats)
            reconstructed_upsampled = F.interpolate(reconstructed, size=(224, 224), mode='bilinear')

            for handle in handles:
                handle.remove()

            rec_loss = criterion(reconstructed_upsampled, imgs) * breast_masks
            rec_loss = rec_loss.sum() / (breast_masks.sum() + 1e-8)
            reg_loss = sum(reg_losses) / max(len(reg_losses), 1) if reg_losses else 0.0
            loss = rec_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            train_loss += loss.item() * imgs.size(0)
            train_samples += imgs.size(0)

        avg_train_loss = train_loss / train_samples if train_samples > 0 else 0
        val_loss = evaluate_mim(val_loader, encoder, decoder, criterion, mask_ratio, block_size)
        test_loss = evaluate_mim(test_loader, encoder, decoder, criterion, mask_ratio, block_size)

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)

        print(f"[Epoch {epoch+1}/] Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}")

        last_ckpt_path = os.path.join(output_dir, "last_checkpoint.pth")
        torch.save({'epoch': epoch + 1, 'encoder_state_dict': encoder.state_dict(), 'decoder_state_dict': decoder.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'train_loss': avg_train_loss, 'val_loss': val_loss, 'test_loss': test_loss}, last_ckpt_path)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_ckpt_path = os.path.join(output_dir, "best_checkpoint.pth")
            torch.save({'epoch': epoch + 1, 'encoder_state_dict': encoder.state_dict(), 'decoder_state_dict': decoder.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'test_loss': best_test_loss}, best_ckpt_path)
            print(f"  [*] New best test loss: {best_test_loss:.4f} => ")

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs+1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs+1), val_losses, label="Val Loss")
    plt.plot(range(1, num_epochs+1), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("L1 Loss")
    plt.title("MIM Reconstruction Loss")
    plt.legend()
    plot_path = os.path.join(output_dir, "mim_loss_curve.png")
    plt.savefig(plot_path)
    print(f"[INFO] MIM Loss curve saved at: {plot_path}")
    if log_mode == "console":
        plt.show()
    else:
        plt.close()

#########################################################
# 6) CLASSIFICATION MODEL & TRAINING
#########################################################

class SwinMIMClassifier(nn.Module):
    def __init__(self, encoder, encoder_dim=1024, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder_dim, num_classes)

    def forward(self, x, mask=None):
        feats = self.encoder.forward_features(x)
        feats = feats.permute(0, 3, 1, 2)
        if mask is not None:
            mask_down = F.interpolate(mask, size=feats.shape[-2:], mode='bilinear', align_corners=False)
            weighted_feats = feats * mask_down
            pooled = weighted_feats.sum(dim=[2, 3]) / (mask_down.sum(dim=[2, 3]) + 1e-8)
            background_feats = feats * (1 - mask_down)
            background_pooled = background_feats.sum(dim=[2, 3]) / ((1 - mask_down).sum(dim=[2, 3]) + 1e-8)
        else:
            pooled = F.adaptive_avg_pool2d(feats, (1, 1)).view(feats.size(0), -1)
            background_pooled = pooled
        logits = self.classifier(pooled)
        return logits, pooled, background_pooled

def compute_class_weights(metadata_df):
    pathology_counts = metadata_df['pathology'].value_counts()
    benign_count = pathology_counts.get('BENIGN', 0) + pathology_counts.get('BENIGN_WITHOUT_CALLBACK', 0)
    malignant_count = pathology_counts.get('MALIGNANT', 0)
    benign_count = max(benign_count, 1)
    malignant_count = max(malignant_count, 1)
    total_samples = benign_count + malignant_count
    weight_benign = total_samples / (2.0 * benign_count)
    weight_malignant = total_samples / (2.0 * malignant_count)
    return torch.tensor([weight_benign, weight_malignant], dtype=torch.float)

def train_classifier(model, train_loader_cls, val_loader_cls, test_loader_cls, metadata_df, lr=1e-4, weight_decay=1e-5, num_epochs=5, output_dir='.', log_mode='console'):
    model = model.cuda()
    class_weights = compute_class_weights(metadata_df).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc = 0.0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        batch_pbar = tqdm(train_loader_cls, desc=f"Cls Epoch {epoch+1}/{num_epochs}",
                          leave=False, file=sys.stdout, mininterval=0.1)
        for imgs, breast_masks, label_dict in batch_pbar:
            imgs = imgs.cuda(non_blocking=True)
            breast_masks = breast_masks.cuda(non_blocking=True)
            labels = convert_pathology_to_label(label_dict['pathology']).cuda()

            handles = []
            reg_losses = []

            def get_hook(name):
                def hook(module, input, output):

                    _, reg_loss = mask_guided_attention_regularization(
                        output[0] if isinstance(output, tuple) else output,
                        breast_masks,
                        block_size=16,
                        lambda_reg=0.1
                    )

                    reg_losses.append(reg_loss)


                    return output

                return hook

            for name, module in model.encoder.named_modules():
                if "layers.3.blocks" in name and "attn" in name:
                    handles.append(module.register_forward_hook(get_hook(name)))

            logits, breast_feats, bg_feats = model(imgs, mask=breast_masks)
            cls_loss = criterion(logits, labels)
            contrastive_loss = F.cosine_similarity(breast_feats, bg_feats).mean()
            reg_loss_total = sum(reg_losses) / max(len(reg_losses), 1) if reg_losses else 0.0
            loss = cls_loss + 0.1 * contrastive_loss + reg_loss_total

            for handle in handles:
                handle.remove()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            train_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += imgs.size(0)

        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, breast_masks, label_dict in val_loader_cls:
                imgs = imgs.cuda(non_blocking=True)
                breast_masks = breast_masks.cuda(non_blocking=True)
                labels = convert_pathology_to_label(label_dict['pathology']).cuda()
                logits, breast_feats, bg_feats = model(imgs, mask=breast_masks)
                cls_loss = criterion(logits, labels)
                contrastive_loss = F.cosine_similarity(breast_feats, bg_feats).mean()
                loss = cls_loss + 0.1 * contrastive_loss
                val_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(f"\n[Epoch {epoch+1}/] Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} || Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print("Validation Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=['Benign', 'Malignant']))

        last_ckpt_path = os.path.join(output_dir, "last_checkpoint_swin_mim_classifier.pth")
        torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_acc': val_acc}, last_ckpt_path)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_ckpt_path = os.path.join(output_dir, "best_checkpoint_swin_mim_classifier.pth")
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_acc': best_val_acc}, best_ckpt_path)
            print(f"  [*] New best Val Acc: {best_val_acc:.4f} => ")

    test_correct, test_total = 0, 0
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, breast_masks, label_dict in tqdm(test_loader_cls, desc="[Test]"):
            imgs = imgs.cuda(non_blocking=True)
            breast_masks = breast_masks.cuda(non_blocking=True)
            labels = convert_pathology_to_label(label_dict['pathology']).cuda()
            logits, _, _ = model(imgs, mask=breast_masks)
            preds = logits.argmax(dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += imgs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = test_correct / test_total
    print(f"\n[Final Test Results] Test Accuracy: {test_acc:.4f}")
    print("Detailed Report:")
    print(classification_report(all_labels, all_preds, target_names=['Benign', 'Malignant']))

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Classification Loss")
    plt.legend()
    loss_plot_path = os.path.join(output_dir, "classification_loss_curve.png")
    plt.savefig(loss_plot_path)
    print(f"[INFO] Classification Loss curve saved at: {loss_plot_path}")
    if log_mode == "console":
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs+1), train_accs, label='Train Acc')
    plt.plot(range(1, num_epochs+1), val_accs, label='Val Acc')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Classification Accuracy")
    plt.legend()
    acc_plot_path = os.path.join(output_dir, "classification_accuracy_curve.png")
    plt.savefig(acc_plot_path)
    print(f"[INFO] Classification Accuracy curve saved at: {acc_plot_path}")
    if log_mode == "console":
        plt.show()
    else:
        plt.close()

#########################################################
# 7) VISUALIZATION
#########################################################

def get_attention_block(model):
    return model.encoder.layers[-1].blocks[-1]

def process_attention_weights(attn_output, img_shape):
    attn = attn_output.detach().cpu()
    feature_magnitudes = torch.norm(attn[0], dim=-1)
    attention_map = (feature_magnitudes - feature_magnitudes.min()) / (feature_magnitudes.max() - feature_magnitudes.min() + 1e-8)
    attention_map = F.interpolate(torch.from_numpy(attention_map.numpy()).unsqueeze(0).unsqueeze(0).float(), size=img_shape[-2:], mode='bicubic', align_corners=False).squeeze().numpy()
    return attention_map

def get_random_sample(dataloader):
    dataset = dataloader.dataset
    if isinstance(dataset, Subset):
        random_idx = random.randint(0, len(dataset) - 1)
        real_idx = dataset.indices[random_idx]
        data = dataset.dataset[real_idx]
    else:
        random_idx = random.randint(0, len(dataset) - 1)
        data = dataset[random_idx]

    if isinstance(data['image'], torch.Tensor):
        image = data['image'].clone().detach().unsqueeze(0).cuda()
    else:
        image = torch.tensor(data['image'], requires_grad=False).unsqueeze(0).cuda()

    if isinstance(data['mask'], torch.Tensor):
        mask = data['mask'].clone().detach().unsqueeze(0).cuda()
    else:
        mask = torch.tensor(data['mask'], requires_grad=False).unsqueeze(0).cuda()

    labels = data.get('labels', {})
    if isinstance(labels, dict):
        for k in labels:
            labels[k] = [labels[k]]

    return image, mask, labels

def visualize_attention(model, dataloader, output_dir='.', log_mode='console'):
    model = model.cuda()
    img, mask, labels = get_random_sample(dataloader)
    true_label = labels.get('pathology', ['Unknown'])[0]
    attn_block = get_attention_block(model)
    attention_values = []
    def hook_fn(module, input, output):
        attention_values.append(output)
    handle = attn_block.register_forward_hook(hook_fn)
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(img, mask=mask)
        pred_class = logits.argmax(dim=1).item()
    handle.remove()
    if not attention_values:
        print("No attention values captured.")
        return
    attention_map = process_attention_weights(attention_values[0], img.shape)
    fig = plt.figure(figsize=(22, 7))
    gs = plt.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.3)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    img_np = img.squeeze().detach().cpu().numpy()
    ax1.imshow(img_np, cmap='gray')
    ax1.set_title(f"Original Image\nLabel: {true_label}")
    ax1.axis('off')
    attention_display = ax2.imshow(attention_map, cmap='inferno')
    ax2.set_title("Raw Attention Map")
    ax2.axis('off')
    plt.colorbar(attention_display, ax=ax2)
    ax3.imshow(img_np, cmap='gray')
    ax3.imshow(attention_map, cmap='inferno', alpha=0.5)
    ax3.set_title(f"Attention Overlay\nPredicted: {'Malignant' if pred_class else 'Benign'}")
    ax3.axis('off')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "attention_map.png")
    plt.savefig(plot_path)
    print(f"[INFO] Attention map saved at: {plot_path}")
    if log_mode == "console":
        plt.show()
    else:
        plt.close()

class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.feature_maps = None
        self.gradient = None
        self.target_layer = target_layer if target_layer is not None else get_target_layer(model)
        self.hooks = []
        self._register_hooks()
        self.is_transformer = self._check_if_transformer()

    def _check_if_transformer(self):
        return any(hasattr(self.model, attr) for attr in ['blocks', 'encoder', 'transformer', 'attention'])

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.feature_maps = output[0] if isinstance(output, tuple) else output
        def full_backward_hook(module, grad_input, grad_output):
            self.gradient = grad_output[0]
        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(full_backward_hook))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def generate_cam(self, input_tensor, mask=None, target_class=None):
        self.model.eval()
        input_tensor = input_tensor.clone()
        input_tensor.requires_grad = True

        logits, _, _ = self.model(input_tensor, mask=mask)
        predicted_class = logits.argmax(dim=1).item()
        target_class = predicted_class if target_class is None else target_class
        self.model.zero_grad()
        target_score = logits[0][target_class]
        target_score.backward(retain_graph=True)

        gradients = self.gradient.detach()
        attention_maps = self.feature_maps.detach()

        gradients = 2 * (gradients - gradients.min()) / (gradients.max() - gradients.min() + 1e-8) - 1
        B, N, D = attention_maps.shape
        attention_scores = torch.matmul(attention_maps, attention_maps.transpose(-2, -1))
        attention_scores = F.softmax(attention_scores / torch.sqrt(torch.tensor(D).float()), dim=-1).mean(dim=0)
        grad_weights = torch.norm(gradients, dim=2)
        grad_weights = F.softmax(grad_weights.mean(dim=0), dim=0)
        cam = attention_scores * grad_weights.view(-1, 1)
        cam = cam.mean(dim=0).view(int(np.sqrt(N)), int(np.sqrt(N)))
        cam = F.relu(cam)
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
            cam = torch.pow(cam, 0.5)
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=input_tensor.shape[-2:], mode='bicubic', align_corners=False).squeeze()
        return cam.detach().cpu().numpy(), predicted_class

def process_gradcam(cam, img_shape):
    if isinstance(cam, torch.Tensor):
        cam = cam.detach().cpu().numpy()

    cam_tensor = torch.from_numpy(cam).float().unsqueeze(0).unsqueeze(0)
    cam_resized = F.interpolate(cam_tensor, size=img_shape[-2:], mode='bicubic', align_corners=False)
    return cam_resized.squeeze().numpy()

def get_target_layer(model):
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
        return model.encoder.layers[-2].blocks[-1].attn
    conv_layers = [module for name, module in model.named_modules() if isinstance(module, torch.nn.Conv2d)]
    return conv_layers[-1] if conv_layers else None

def visualize_gradcam(model, dataloader, output_dir='.', log_mode='console', target_layer=None):
    model = model.cuda()
    model.eval()
    img, mask, labels = get_random_sample(dataloader)
    true_label = labels.get('pathology', ['Unknown'])[0]

    grad_cam = GradCAM(model, target_layer)

    cam, pred_class = grad_cam.generate_cam(img, mask=mask)

    cam_processed = process_gradcam(cam, img.shape)

    fig = plt.figure(figsize=(22, 7))
    gs = plt.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.3)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])

    img_np = img.detach().squeeze().cpu().numpy()

    ax1.imshow(img_np, cmap='gray')
    ax1.set_title(f"Original Image\nLabel: {true_label}")
    ax1.axis('off')

    gradcam_display = ax2.imshow(cam_processed, cmap='RdYlBu_r', vmin=0, vmax=1)
    ax2.set_title("GRAD-CAM Heatmap")
    ax2.axis('off')
    plt.colorbar(gradcam_display, ax=ax2)

    ax3.imshow(img_np, cmap='gray')
    overlay = ax3.imshow(cam_processed, cmap='RdYlBu_r', alpha=0.7, vmin=0, vmax=1)
    ax3.set_title(f"GRAD-CAM Overlay\nPredicted: {'Malignant' if pred_class else 'Benign'}")
    ax3.axis('off')
    plt.colorbar(overlay, ax=ax3)

    try:
        plt.tight_layout()
    except Warning:
        print("[WARNING] Tight layout might not work correctly with this figure arrangement")

    plot_path = os.path.join(output_dir, "gradcam_map.png")
    plt.savefig(plot_path)
    print(f"[INFO] Grad-CAM map saved at: {plot_path}")

    if log_mode == "console":
        plt.show()
    else:
        plt.close()

    grad_cam.remove_hooks()
#########################################################
# 8) PARSE METADATA
#########################################################

def process_dataframe(df, dcm_files, is_mass=True):
    prefix = "Mass-Training" if is_mass else "Calc-Training"
    density_col = 'breast_density' if is_mass else 'breast density'
    data = []
    for _, row in df.iterrows():
        patient_id = row['patient_id'].split('_')[1]
        breast = row['left or right breast']
        view = row['image view']
        folder_pattern = f"{prefix}_P_{patient_id}_{breast}_{view}"
        full_image_path = find_full_mammogram(dcm_files, folder_pattern)
        if full_image_path:
            item = {
                'image_path': full_image_path[0],
                'label': 1 if is_mass else 0,
                'pathology': row['pathology'],
                'subtlety': row['subtlety'],
                'breast_density': row[density_col],
                'assessment': row['assessment'],
                'abnormality_type': row['abnormality type'],
                'patient_id': row['patient_id'],
                'breast': breast,
                'view': view
            }
            if is_mass:
                item['mass_shape'] = row['mass shape']
                item['mass_margins'] = row['mass margins']
            else:
                item['calc_type'] = row['calc type']
                item['calc_distribution'] = row['calc distribution']
            data.append(item)
    return data

def parse_ddsm_metadata(root_dir, cache_dir_cls, dcm_files, dicom_path, force_recompute=False):
    # Save metadata in the same directory as dicom_path
    # if not dicom_path:
    # #     cache_file = os.path.join(root_dir, 'metadata_df.pkl')
    # # else:
    #     # dicom_dir = os.path.dirname(dicom_path)
    #     dcm_files = glob.glob(os.path.join(root_dir, "**", "*.dcm"), recursive=True)
        # cache_file = os.path.join(root_dir, 'metadata_df.pkl')

    # if not force_recompute and os.path.exists(cache_file):
    #     with open(cache_file, 'rb') as f:
    #         metadata_df = pickle.load(f)
    #     print(f"[INFO] Loaded metadata with {len(metadata_df)} records from: {cache_file}")
    #     return metadata_df

    mass_train_csv = os.path.join(root_dir, 'mass_case_description_train_set.csv')
    mass_test_csv = os.path.join(root_dir, 'mass_case_description_test_set.csv')
    calc_train_csv = os.path.join(root_dir, 'calc_case_description_train_set.csv')
    calc_test_csv = os.path.join(root_dir, 'calc_case_description_test_set.csv')
    mass_train = pd.read_csv(mass_train_csv)
    mass_test = pd.read_csv(mass_test_csv)
    calc_train = pd.read_csv(calc_train_csv)
    calc_test = pd.read_csv(calc_test_csv)
    all_data = []
    all_data.extend(process_dataframe(mass_train, dcm_files, is_mass=True))
    all_data.extend(process_dataframe(mass_test, dcm_files, is_mass=True))
    all_data.extend(process_dataframe(calc_train, dcm_files, is_mass=False))
    all_data.extend(process_dataframe(calc_test, dcm_files, is_mass=False))
    metadata_df = pd.DataFrame(all_data)
    cache_file = os.path.join(cache_dir_cls, 'metadata_df_cls.pkl')
    with open(cache_file, 'wb') as f:
        pickle.dump(metadata_df, f)
    print(f"[INFO] Processed and cached metadata with {len(metadata_df)} records to {cache_file}")
    return metadata_df

#########################################################
# 9) MAIN
#########################################################

def main():
    parser = argparse.ArgumentParser(description="CBIS-DDSM MIM & Classification + Visualization")
    parser.add_argument('--root_dir', type=str, required=True, help="Path to the directory containing CBIS-DDSM CSV files.")
    parser.add_argument('--dicom_path', type=str, default="", help="Path to a text file containing DICOM paths.")
    parser.add_argument('--output_dir', type=str, default='./outputs', help="Directory to save checkpoints/results")
    parser.add_argument('--cache_dir', type=str, default='./cache', help="Directory to save cached MIM files")
    parser.add_argument('--cache_dir_cls', type=str, default="", help="Directory to save cached CLS files")
    parser.add_argument('--mim_epochs', type=int, default=3, help="Number of epochs for MIM pre-training")
    parser.add_argument('--cls_epochs', type=int, default=3, help="Number of epochs for classification training")
    parser.add_argument('--missing_label_percentage', type=int, default=100, help="Percentage of labeled training data to use")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for dataloaders")
    parser.add_argument('--num_workers', type=int, default=2, help="Number of workers for dataloaders")
    parser.add_argument('--train_mim', action='store_true', help="Train MIM model from scratch")
    parser.add_argument('--train_cls', action='store_true', help="Train classification model using MIM encoder")
    parser.add_argument('--visualize', action='store_true', help="Perform visualization")
    parser.add_argument('--force_recompute', action='store_true', help="Force recomputation of cached files")
    parser.add_argument('--force_recompute_cls', action='store_true', help="Force recomputation of cached CLS files")
    parser.add_argument('--log_mode', type=str, choices=['console', 'file'], default='console', help="Log mode")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.log_mode == 'file':
        log_path = os.path.join(args.output_dir, "training_log.txt")
        sys.stdout = open(log_path, 'w', buffering=1)
        matplotlib.use('Agg')
        print(f"[INFO] Logging to file: {log_path}")

    if args.dicom_path:
        with open(args.dicom_path, "r") as f:
            dcm_files = [line.strip() for line in f]
        print(f"[INFO] Loaded {len(dcm_files)} DICOM file paths from '{args.dicom_path}'.")
    else:
        dcm_files = glob.glob(os.path.join(args.root_dir, "**", "*.dcm"), recursive=True)
        print(f"[INFO] Found {len(dcm_files)} DICOM files by scanning '{args.root_dir}'.")
        with open(args.dicom_path, "w") as f:
          for path in dcm_files:
              f.write(path + "\n")

    # if len(dcm_files) > 0:
    #     full_mammogram_files = find_full_mammogram(dcm_files, None)
    #     if full_mammogram_files:
    #         debug_transform_sequence(full_mammogram_files[0], output_dir=args.output_dir)
    #     else:
    #         print("[WARNING] No full mammogram images found.")
    # else:
    #     print("[WARNING] No DICOM files found.")

    mim_train_loader, mim_val_loader, mim_test_loader = create_mim_dataloaders(
        args.root_dir, dcm_files, batch_size=args.batch_size, num_workers=args.num_workers, cache_dir=args.cache_dir, force_recompute=args.force_recompute
    )
    if args.cache_dir_cls and not args.force_recompute_cls:
        cache_file = os.path.join(args.cache_dir_cls, 'metadata_df_cls.pkl')
        with open(cache_file, 'rb') as f:
            metadata_df = pickle.load(f)
    else:
        metadata_df = parse_ddsm_metadata(
            args.root_dir, args.cache_dir_cls, dcm_files, args.dicom_path, force_recompute=args.force_recompute
        )
    cls_train_loader, cls_val_loader, cls_test_loader = create_partial_dataloaders(
        metadata_df, batch_size=args.batch_size, num_workers=args.num_workers, missing_label_percentage=args.missing_label_percentage
    )

    encoder = timm.create_model("swin_base_patch4_window7_224", in_chans=1, pretrained=True, num_classes=0).cuda()

    if args.train_mim:
        print("\n[INFO] Starting MIM Training ...")
        decoder = MaskedDecoder(in_dim=1024, out_ch=1).cuda()
        train_mim(encoder=encoder, decoder=decoder, train_loader=mim_train_loader, val_loader=mim_val_loader, test_loader=mim_test_loader, num_epochs=args.mim_epochs, output_dir=args.output_dir, log_mode=args.log_mode)
    else:
        best_ckpt_path = os.path.join(args.output_dir, "best_checkpoint.pth")
        if os.path.exists(best_ckpt_path):
            ckpt = torch.load(best_ckpt_path, map_location='cuda')
            encoder.load_state_dict(ckpt['encoder_state_dict'])
            print(f"[INFO] Loaded MIM checkpoint from {best_ckpt_path}")
        else:
            print("[WARNING] No MIM checkpoint found; using encoder as-is.")

    if args.train_cls:
        print("\n[INFO] Starting Classification Training ...")
        model_cls = SwinMIMClassifier(encoder=encoder, encoder_dim=1024, num_classes=2).cuda()
        train_classifier(model=model_cls, train_loader_cls=cls_train_loader, val_loader_cls=cls_val_loader, test_loader_cls=cls_test_loader, metadata_df=metadata_df, num_epochs=args.cls_epochs, output_dir=args.output_dir, log_mode=args.log_mode)
    else:
        best_cls_ckpt = os.path.join(args.output_dir, "best_checkpoint_swin_mim_classifier.pth")
        if os.path.exists(best_cls_ckpt):
            print(f"[INFO] Found classification checkpoint: {best_cls_ckpt}")
        else:
            print("[INFO] No classification checkpoint found.")

    if args.visualize:
        print("\n[INFO] Visualization ...")
        model_cls = SwinMIMClassifier(encoder=encoder, encoder_dim=1024, num_classes=2).cuda()
        best_cls_ckpt = os.path.join(args.output_dir, "best_checkpoint_swin_mim_classifier.pth")
        if os.path.exists(best_cls_ckpt):
            ckpt = torch.load(best_cls_ckpt, map_location='cuda')
            model_cls.load_state_dict(ckpt['model_state_dict'])
            print(f"[INFO] Loaded classification checkpoint from {best_cls_ckpt}")
        visualize_attention(model_cls, cls_test_loader, output_dir=args.output_dir, log_mode=args.log_mode)
        visualize_gradcam(model_cls, cls_test_loader, output_dir=args.output_dir, log_mode=args.log_mode)

    if args.log_mode == 'file':
        sys.stdout.close()

if __name__ == "__main__":
    main()
