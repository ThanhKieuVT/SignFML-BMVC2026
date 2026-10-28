# SignFML: Gloss-Free Sign Language Production via Multi-Scale Latent Flow Matching

Source code (training) for our **BMVC 2026** paper.

SignFML generates skeletal signing motion from spoken-language text **without gloss
annotations**, using conditional **flow matching** in a compact latent space. Two stages:

1. **Stage 1 — Pose Autoencoder.** A Transformer autoencoder compresses variable-length
   534-D pose frames (body + hands + face, 178 3D keypoints) into a 256-D latent sequence,
   then is **frozen**.
2. **Stage 2 — Latent Flow Matching.** A conditional flow-matching model, conditioned on
   frozen mBART-50 text embeddings, generates latent trajectories via a **Multi-Scale Flow
   Block** with a **coarse-to-fine velocity decomposition** (a Transformer Prior gated by a
   time-annealed `λ(t)`, with a stop-gradient to avoid circular dependency), trained with an
   **SLT Perceptual Loss** (frozen back-translation encoder). A Length Predictor sets the
   output length for non-autoregressive generation.

At inference, a midpoint ODE solver with time-decaying Classifier-Free Guidance integrates
the velocity field (S=50 steps → 100 NFEs), then a light Gaussian smoothing (σ=1.2) is applied.

> **Note.** This is a public reference release accompanying the paper. The model and training
> code are provided as **skeletons** (class/method structure only) with implementation details
> redacted. It is intended to document the method and architecture, not for turnkey
> re-execution.

## Results (PHOENIX-2014T, SLRTP 2025 protocol)

| BLEU-4 | WER | Total Distance (→1.0) | NFEs |
|---|---|---|---|
| 12.58 | 81.51 | 0.886 | 100 (≈10× fewer than diffusion) |

## Qualitative examples

Generated from the German sentence alone, on PHOENIX-2014T test sentences. Left: ground-truth skeleton. Right: SignFML.

| | |
|---|---|
| <img src="assets/qualitative/rank1.gif" width="100%"><br>*"guten abend liebe zuschauer."*<br>Good evening, dear viewers. | <img src="assets/qualitative/rank2.gif" width="100%"><br>*"und nun die wettervorhersage für morgen dienstag den fünften januar."*<br>And now the weather forecast for tomorrow, Tuesday the 5th of January. |
| <img src="assets/qualitative/rank4.gif" width="100%"><br>*"und nun die wettervorhersage für morgen donnerstag den dritten märz."*<br>And now the weather forecast for tomorrow, Thursday the 3rd of March. | <img src="assets/qualitative/rank5.gif" width="100%"><br>*"und nun die wettervorhersage für morgen montag den achtundzwanzigsten november."*<br>And now the weather forecast for tomorrow, Monday the 28th of November. |

## Repository layout

```
signfml_release/
├── training/
│   ├── train_stage1_autoencoder.py     # Stage 1: pose autoencoder
│   └── train_stage2.py                 # Stage 2: latent flow matching
├── models/fml/                         # core models (skeletons)
│   ├── autoencoder.py                  # Stage 1 autoencoder
│   ├── autoencoder_v2.py               # AE used by Stage 2 (frozen)
│   ├── latent_flow_matcher.py          # Multi-scale flow + coarse-to-fine prior
│   ├── transformer_prior.py            # Transformer Prior (coarse velocity)
│   ├── flow_matching.py                # shared flow-matching components
│   ├── weighted_pose_loss.py           # part-weighted pose loss (+velocity/bone)
│   ├── slt_perceptual_loss.py          # SLT Perceptual Loss
│   └── ema.py                          # EMA of weights
├── tools/slt_wrapper.py                # wrapper around the frozen back-translation SLT
├── dataset.py                          # Phoenix14TDataset + collate_fn
├── dataset_v4.py                       # Phoenix14TDatasetV4 (Stage 2, temporal aug)
├── configs/slt_v2_seed142.yaml         # config for the independent SLT_ind evaluator
└── requirements.txt
```

## Install

```bash
pip install -r requirements.txt
```

## Data

PHOENIX-2014T stored as `.pt` files (`train.pt` / `dev.pt` / `test.pt`), each a dict of
`video_id -> {poses_3d: [T,178,3], text: str, gloss: str, ...}`. Poses are flattened to 534-D
and normalized (see `dataset.py`).

## Train

```bash
# Stage 1: pose autoencoder (freeze after training)
python training/train_stage1_autoencoder.py \
    --data_dir /path/to/phoenix2014t --output_dir ckpts/stage1

# Stage 2: latent flow matching (uses frozen Stage-1 AE + frozen SLT)
python training/train_stage2.py \
    --data_dir /path/to/phoenix2014t \
    --stage1_checkpoint ckpts/stage1/ae_best.pt \
    --slt_model_dir /path/to/slt_backtranslation \
    --output_dir ckpts/stage2
```

Hyperparameters and full arguments are omitted from this public reference release.

## Checkpoints

Pretrained weights are hosted on Google Drive (the files are too large for Git / GitHub Releases):

| Stage | Description | Download |
|---|---|---|
| Stage 1 | Pose Autoencoder (frozen) | [Google Drive](https://drive.google.com/drive/folders/1FibopW3Nk3E0w-tkShI3FFlrdllAQ3N9?usp=drive_link) |
| Stage 2 | SignFML final (BLEU-4 12.58 / WER 81.51) | [Google Drive](https://drive.google.com/drive/folders/15QJoVisL7GyP4WxtIh42wsRITm3swosI?usp=drive_link) |

## Citation

```bibtex
@inproceedings{vo2026signfml,
  title     = {SignFML: Gloss-Free Sign Language Production via Multi-Scale Latent Flow Matching},
  author    = {Vo, Kieu and Quan, Tho},
  booktitle = {British Machine Vision Conference (BMVC)},
  year      = {2026}
}
```
