# Mitigating Hallucinations in Mammography: A Spectral-Regularized Attention Framework with Breast-Aware Masked Pretraining

This repository implements a novel framework for mammography classification that mitigates "hallucinations" - the tendency of models to focus on irrelevant background patterns rather than clinically significant features. The approach combines a hierarchical Swin Transformer backbone with spectral regularization and breast-aware masked pretraining.

## Technical Overview

This framework addresses common challenges in mammogram classification through four key mechanisms:

1. **Enhanced Attention with Spectral Regularization**: Applying spectral norm penalties to stabilize and sharpen the model's attention on relevant breast tissue.

2. **Breast-Aware Masking for MIM Pretraining**: A self-supervised pretraining strategy that masks only within breast regions to focus representation learning on true tissue signals. By forcing the model to reconstruct only breast tissue content, the learned representations focus on clinically relevant structures and suppress background "noise."

3. **Transformer-Based Classification**: Employing a hierarchical Swin Transformer backbone for multi-scale feature extraction in high-resolution mammograms. This structure enables linear complexity with image size and strong multi-scale feature representation.

4. **Attention-Based Explainability**: Providing interpretability via Grad-CAM and attention attribution maps highlighting model decision regions.

The model architecture incorporates:
- Breast segmentation to identify regions of interest
- Spectral norm regularization to constrain attention mechanisms
- Contrastive learning to separate breast and background features
- Multi-head self-attention with window-based processing

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--root_dir` | Required | Path to directory containing CBIS-DDSM CSV files |
| `--dicom_path` | "" | Path to text file containing list of DICOM file paths |
| `--output_dir` | ./outputs | Directory to save checkpoints, results, and visualizations |
| `--cache_dir` | ./cache | Directory to save cached MIM files |
| `--cache_dir_cls` | "" | Directory to save cached classification files |
| `--mim_epochs` | 3 | Number of epochs for MIM pre-training |
| `--cls_epochs` | 3 | Number of epochs for classification training |
| `--missing_label_percentage` | 100 | Percentage of labeled training data to use |
| `--batch_size` | 4 | Batch size for data loaders |
| `--num_workers` | 2 | Number of worker processes for data loading |
| `--train_mim` | False | Flag to train MIM model from scratch |
| `--train_cls` | False | Flag to train classification model |
| `--visualize` | False | Flag to generate attention maps and Grad-CAM visualizations |
| `--force_recompute` | False | Flag to force recomputation of cached MIM files |
| `--force_recompute_cls` | False | Flag to force recomputation of cached classification files |
| `--log_mode` | console | Controls logging behavior: 'console' for interactive display, 'file' for saving to output directory |

## Execution Commands

### MIM (Masked Image Modeling) Pre-training

```bash
python script.py \
  --root_dir /path/to/CBIS-DDSM_directory \
  --dicom_path /path/to/dicom_files.txt \
  --output_dir ./outputs \
  --cache_dir ./cache \
  --mim_epochs 5 \
  --train_mim \
  --batch_size 4 \
  --num_workers 2 \
  --log_mode console
