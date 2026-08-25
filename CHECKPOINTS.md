# Checkpoints

We release the two model checkpoints needed to reproduce SignFML.
(Weights are hosted externally; see the release links.)

| Checkpoint | Stage | Description | Paper reference |
|---|---|---|---|
| `stage1/ae_best.pt` | Stage 1 | Frozen Unified Pose Autoencoder (534-D ↔ 256-D latent). Near-lossless reconstruction (DTW-MJE 0.0024). | Sec. Implementation Details |
| `stage2/best_bleu_model.pt` | Stage 2 | Final SignFML latent flow matcher (multi-scale flow + coarse-to-fine prior + SLT loss). BLEU-4 12.58 / WER 81.51. Includes the Length Predictor. | Table 1 (main result) |

## Loading example

```python
import torch
from models.fml.autoencoder_v2 import UnifiedPoseAutoencoder
from models.fml.latent_flow_matcher_v11_24mar import LatentFlowMatcherV11

# Stage 1 (frozen autoencoder)
ae = UnifiedPoseAutoencoder(latent_dim=256)
ae.load_state_dict(torch.load("stage1/ae_best.pt", map_location="cpu")["model_state_dict"], strict=False)
ae.eval()

# Stage 2 (flow matcher)
flow = LatentFlowMatcherV11()
flow.load_state_dict(torch.load("stage2/best_bleu_model.pt", map_location="cpu")["model_state_dict"], strict=False)
flow.eval()
```

> Exact constructor args and state-dict keys follow the training scripts in `training/`.
> Inspect a checkpoint with `torch.load(path, map_location="cpu").keys()`
> (weights are typically under `model_state_dict` / `ema_state_dict`).
