# Mitigating Hallucinations in Mammography: A Spectral-Regularized Attention Framework with Breast-Aware Masked Pretraining

## Technical Architecture

This framework addresses challenges in mammography classification through a three-phase approach.

### Phase 1: Breast-Aware Masked Image Modeling (MIM) Pretraining

The framework performs self-supervised pretraining using a breast-aware Masked Image Modeling (MIM) approach. A binary breast mask, derived from a segmentation model, is applied to ensure the model focuses on the breast region during pretraining.

#### Breast Segmentation and Mask Generation

We define a binary mask $M(x) \in \{0,1\}^{H \times W}$ for each mammogram $x$ using a sequence of image processing operations (CLAHE, Otsu’s thresholding, and morphological operations).

#### Breast-Aware Masking Strategy

We partition the image into non-overlapping blocks, identify blocks overlapping the breast region, and randomly mask a subset. Mathematically, eligible breast blocks are defined as:

$$B = \bigl\{ p_i \mid \text{avg\_pool2d}(M(x), b, b)(r_i, c_i) > \tau \bigr\},$$

where \(\tau\) is a threshold parameter (default \(\tau = 0.5\)).

#### Reconstruction Objective

The loss function is:

$$\mathcal{L}_{\mathrm{MIM}}
= \frac{1}{B} \sum_{b=1}^{B}
  \frac{\bigl\| f_{\theta}\bigl(\tilde{x}_b\bigr) - x_b \bigr\|_{1} \,\cdot\, M(x_b)}
       {\sum_{i,j} M(x_b)_{i,j} + \epsilon},$$

where \(\tilde{x}\) is the masked image, and \(\epsilon = 10^{-8}\) prevents division by zero.

#### Spectral Norm Regularization

We apply spectral regularization to the query and key projection matrices in self-attention layers:

$$\mathcal{L}_{\mathrm{SN,base}}
= \beta \sum_{l=1}^{L}
  \Bigl[
    \sigma_{\max}\bigl(W^{Q}_{(l)}\bigr)^2
    + \sigma_{\max}\bigl(W^{K}_{(l)}\bigr)^2
  \Bigr],$$

where \(\sigma_{\max}(W)\) denotes the largest singular value of matrix \(W\).

### Phase 2: Spectral-Regularized Fine-Tuning for Classification

We fine-tune the pretrained Swin Transformer encoder for binary classification (benign vs. malignant).

#### Classification Architecture

The model processes input through hierarchical stages with progressively increasing channel dimensions and decreasing spatial resolution, producing a feature map:

$$F \,\in\, \mathbb{R}^{B \times C \times H' \times W'}.$$

#### Mask-Weighted Pooling

To focus on breast tissue regions:

$$z
= \frac{\displaystyle\sum_{i=0}^{H'-1} \sum_{j=0}^{W'-1}
       F(:,:,i,j)\,\cdot\,M_{\mathrm{down}}(i,j)}
       {\displaystyle\sum_{i=0}^{H'-1} \sum_{j=0}^{W'-1}
       M_{\mathrm{down}}(i,j) + \epsilon},$$

where \(M_{\mathrm{down}}\) is the breast mask downsampled to match feature dimensions.

#### Contrastive Loss

We enforce separation between breast tissue and background representations:

$$
\mathcal{L}_{\mathrm{contrast}}
= \frac{1}{B} \sum_{b=1}^{B}
  \cos\bigl(f^b_{\mathrm{breast}},\, f^b_{\mathrm{bg}}\bigr),
$$

where $\cos(a,b) = \frac{a \cdot b}{||a|| \cdot ||b||}$ is the cosine similarity between vectors.

minimizing similarity between tissue and background features.

#### Total Fine-Tuning Loss

The complete objective integrates cross-entropy, spectral regularization, and contrastive terms:

#### Spectral Norm for Attention Masks

We also apply spectral regularization to the mask attention matrices:

$$\mathcal{L}_{\mathrm{SN,mask}}
= \gamma \sum_{l=1}^{L}
  \sigma_{\max}\bigl(A_{(l)}\bigr)^2,$$

where $A_{(l)}$ is the attention map at layer $l$.

### Phase 3: Attention-Based Explainability

The framework provides visual explanations of the model’s decisions.

#### Attention Maps

We extract and visualize attention weights from the final self-attention layer:

$$
\overline{A}
= \frac{1}{h} \sum_{i=1}^{h} A_i,
$$

where \(h\) is the number of attention heads.

#### Gradient-weighted Class Activation Mapping (Grad-CAM)

We implement Grad-CAM for our transformer architecture, computing class-specific gradient weights:

$$
\alpha_k^c
= \frac{1}{H'W'}
  \sum_{i=0}^{H'-1} \sum_{j=0}^{W'-1}
  \frac{\partial y^c}{\partial F_{k}(:,:,i,j)},
$$

where \(y^c\) represents the score for class \(c\).

The spectral-regularized attention maps demonstrate concentrated activation on anatomically plausible malignant findings, while maintaining minimal attention in background areas, confirming the model’s focus on diagnostically relevant tissue regions.


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
