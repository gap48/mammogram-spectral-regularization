# Setup and Usage Instructions

## Required Directory Structure

Before running the code, ensure the following directory structure:

```markdown
project_root/
├── script.py                                 # Main script
├── run_script.py                             # Optional: Colab execution script
├── manifest-ZkhPvrLo5216730872708713142/
│   └── CBIS-DDSM/                           # Dataset directory
│       ├── mass_case_description_train_set.csv
│       ├── mass_case_description_test_set.csv
│       ├── calc_case_description_train_set.csv
│       ├── calc_case_description_test_set.csv
│       └── full mammogram images/
│           ├── Mass-Training_P_00001_LEFT_CC/
│           ├── Mass-Training_P_00001_LEFT_MLO/
│           └── ...
├── dcm_files.txt                             # Optional: List of DICOM paths
└── outputs/                                  # Created automatically
    ├── checkpoints/
    ├── plots/
    ├── visualizations/
    └── training_log.txt
```
## Training Modes
### MIM (Masked Image Modeling) Pre-training
To train the MIM model from scratch:
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
```

### Classification Training
To train the classification model using a pre-trained MIM encoder:
```bash
python script.py \
  --root_dir /path/to/CBIS-DDSM_directory \
  --dicom_path /path/to/dicom_files.txt \
  --output_dir ./outputs \
  --cache_dir_cls ./cache_cls \
  --cls_epochs 10 \
  --train_cls \
  --batch_size 4 \
  --num_workers 2 \
  --log_mode console
```

### End-to-End Training
To perform both MIM pre-training and classification training in sequence:
```bash
python script.py \
  --root_dir /path/to/CBIS-DDSM_directory \
  --dicom_path /path/to/dicom_files.txt \
  --output_dir ./outputs \
  --cache_dir ./cache \
  --cache_dir_cls ./cache_cls \
  --mim_epochs 5 \
  --cls_epochs 10 \
  --train_mim \
  --train_cls \
  --batch_size 4 \
  --num_workers 2 \
  --log_mode console
```

### Visualization Only
To generate attention maps and Grad-CAM visualizations using a previously trained model:
```bash
python script.py \
  --root_dir /path/to/CBIS-DDSM_directory \
  --dicom_path /path/to/dicom_files.txt \
  --output_dir ./outputs \
  --visualize \
  --log_mode console
```

### Output Structure

```markdown
outputs/
├── MIM Checkpoints/
│   ├── best_checkpoint.pth                         # MIM model weights with best validation loss
│   └── last_checkpoint.pth                         # MIM model weights from the last training epoch
│
├── Classification Checkpoints/
│   ├── best_checkpoint_swin_mim_classifier.pth     # Classification model weights with best validation accuracy
│   └── last_checkpoint_swin_mim_classifier.pth     # Classification model weights from the last epoch
│
├── MIM Training Plots/
│   ├── mim_loss_curve.png                          # Loss curves for MIM training
│   └── mim_visual_epoch{epoch}_batch{batch}.png    # Visualizations of masked images during training
│
├── Classification Training Plots/
│   ├── classification_loss_curve.png               # Loss curves for classification training
│   └── classification_accuracy_curve.png           # Accuracy curves for classification training
│
├── Visualizations/
│   ├── attention_map.png                           # Attention map overlay for breast cancer classification
│   └── gradcam_map.png                             # Grad-CAM heatmap and overlay for interpretability
│
└── Logs/
    └── training_log.txt                            # (If log_mode=file) Detailed logs of training progress
```
