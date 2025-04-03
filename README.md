# Mammography Classification with Spectral-Regularized Attention and Breast-Aware Pretraining

This framework enhances mammography classification by mitigating model "hallucinations" through a three-phase process:

1. **Breast-Aware MIM Pretraining**: Uses a Swin Transformer and convolutional decoder for self-supervised learning, emphasizing breast tissue over background features and contrastive loss to capture subtle anatomical cues
2. **Spectral-Regularized Fine-Tuning**: Fine-tunes for benign vs. malignant classification with mask-weighted pooling and attention modulation between breast tissue area and background region
3. **Attention-Based Explainability**: Provides interpretable predictions using attention maps and Grad-CAM.

## Technical Architecture

- **Backbone**: Swin Transformer adapted for single-channel mammograms.
- **Pretraining**: Breast-aware masked image modeling with spectral norm regularization.
- **Fine-Tuning**: Combines cross-entropy, spectral regularization, and contrastive loss.
- **Interpretability**: Visualizes clinically relevant features.

### Phase 1: Breast-Aware MIM Pretraining

- **Mask Generation**: A binary mask $`M(x) \in \{0,1\}^{H \times W}`$ where $`M(x)_{u,v} = 1`$ for breast tissue pixels, else 0.
- **Masking Strategy**: Masks blocks where the average mask value $`\text{avg-pool2d}(M(x), b, b)(r, c) > \tau`$
- **Reconstruction Loss**: 
  $\mathcal{L}_{\text{MIM}} = \frac{1}{B} \sum_{b=1}^{B} \frac{\left\| f_\theta(\tilde{x}_b) - x_b \right\|_1 \cdot M(x_b)}{\sum M(x_b) + \epsilon}$
  where $`\tilde{x}_b`$ is the masked input, $`f_\theta`$ is the reconstruction function, and $`\epsilon`$ prevents division by zero.
- **Spectral Regularization**: 
  $$
  \mathcal{L}_{\text{SN,base}} = \beta \sum_{l=1}^{L} \left[ \sigma_{\max}(W_{(l)}^Q)^2 + \sigma_{\max}(W_{(l)}^K)^2 \right]
  $$
  applied to query ($`W^Q`$) and key ($`W^K`$) matrices in self-attention layers.
- **Mask-Guided Regularization**: 
  $$
  \mathcal{L}_{\text{SN,mask}} = \lambda_{\text{reg}} \cdot (1 - \text{breast-ratio}) \cdot \text{attn-avg}
  $$
  where $`\text{breast-ratio}`$ is the proportion of breast tissue, and $`\text{attn-avg}`$ is the average attention score.
- **Total Pretraining Loss**: 
  $$
  \mathcal{L}_{\text{pretrain}} = \mathcal{L}_{\text{MIM}} + \mathcal{L}_{\text{SN,base}} + \mathcal{L}_{\text{SN,mask}}
  $$

### Phase 2: Spectral-Regularized Fine-Tuning

- **Mask-Weighted Pooling**: 
  $$
  \mathbf{z} = \frac{\sum_{i,j} F(i,j) \cdot M_{\text{down}}(i,j)}{\sum_{i,j} M_{\text{down}}(i,j) + \epsilon}
  $$
  where $`F`$ is the feature map, and $`M_{\text{down}}`$ is the downsampled mask.
- **Classification Logits**: 
  $$
  \text{logits} = W_{\text{cls}} \mathbf{z} + \mathbf{b}_{\text{cls}}
  $$
- **Contrastive Loss**: 
  $$
  \mathcal{L}_{\text{contrast}} = \frac{1}{B} \sum_{b=1}^{B} \cos(\mathbf{f}_{\text{breast}}^b, \mathbf{f}_{\text{bg}}^b)
  $$
  where $`\mathbf{f}_{\text{breast}}`$ and $`\mathbf{f}_{\text{bg}}`$ are embeddings from breast and background regions.
- **Total Fine-Tuning Loss**: 
  $$
  \mathcal{L}_{\text{final}} = \mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{SN,base}} + \mathcal{L}_{\text{SN,mask}} + \alpha \mathcal{L}_{\text{contrast}}
  $$
  where $`\mathcal{L}_{\text{CE}}`$ is the cross-entropy loss, and $`\alpha`$ balances the contrastive term.

### Phase 3: Attention-Based Explainability

- **Attention Maps**: 
  $$
  \overline{A} = \frac{1}{h} \sum_{i=1}^{h} A_i
  $$
  averaged across $`h`$ heads, then upsampled and normalized.
- **Grad-CAM**: 
  $$
  \alpha_k^c = \frac{1}{H' W'} \sum_{i,j} \frac{\partial y^c}{\partial F_k(i,j)}, \quad L_{\text{Grad-CAM}}^c(i,j) = \text{ReLU}\left( \sum_{k} \alpha_k^c F_k(i,j) \right)
  $$
  computed for class $`c`$, then upsampled to the input resolution.

## Key Features

- **Spectral Regularization**: Stabilizes attention to focus on relevant features.
- **Breast-Aware MIM**: Prioritizes breast tissue during pretraining.
- **Explainability**: Ensures clinical reliability via attention visualization.

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
