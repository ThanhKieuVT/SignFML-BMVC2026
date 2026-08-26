#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 Training Script V11 (24 Mar) — Multi-Scale Flow + Adaptive ODE + Dynamic CFG
=====================================================================================
Author: Kieu Vo
Date: 24 Mar 2026

Based on train_stage2_v5_28feb.py with:
- LatentFlowMatcherV11 (Multi-Scale, Temporal Bias, Duration-Aware Length)
- Partial warm-start from V10/V5 checkpoint (text_proj + ssm_prior reused)
- 3 LR groups: warm-started params (low), new flow_block (high), new length_predictor (high)
"""

import os
import sys
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler

def get_raw_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.fml.latent_flow_matcher import LatentFlowMatcherV11
from models.fml.autoencoder_v2 import UnifiedPoseAutoencoder
from models.fml.ema import EMA
from models.fml.weighted_pose_loss import SignLanguageLoss
from models.fml.slt_perceptual_loss import SLTPerceptualLoss
from dataset_v4 import Phoenix14TDatasetV4 as SignLanguageDataset, collate_fn
from tools.slt_wrapper import SLTWrapper

# =============================================================================
# REUSE: CachedFeaturesDataset, make_cached_collate_fn, build_text_feature_cache
# Import from V5 training script
# =============================================================================

class CachedFeaturesDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, cache: dict):
        self.base = base_dataset
        self.cache = cache
        if hasattr(base_dataset, 'stats'):
            self.stats = base_dataset.stats

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        vid_id = item['video_id']
        if vid_id in self.cache:
            item['raw_text_feats'] = self.cache[vid_id]
        return item


def make_cached_collate_fn(base_collate_fn):
    def cached_collate(batch):
        has_cache = 'raw_text_feats' in batch[0]
        if has_cache:
            raw_feats_list = [item.pop('raw_text_feats') for item in batch]
            max_len = max(f.shape[0] for f in raw_feats_list)
            D = raw_feats_list[0].shape[1]
            padded = torch.zeros(len(raw_feats_list), max_len, D)
            cache_mask = torch.zeros(len(raw_feats_list), max_len, dtype=torch.bool)
            for i, f in enumerate(raw_feats_list):
                padded[i, :f.shape[0]] = f
                cache_mask[i, :f.shape[0]] = True
        collated = base_collate_fn(batch)
        if has_cache:
            collated['raw_text_feats'] = padded
            collated['attention_mask'] = cache_mask
            collated['text_tokens'] = collated['text_tokens'][:, :max_len]
        return collated
    return cached_collate


@torch.no_grad()
def build_text_feature_cache(dataset, model, device, cache_path=None, batch_size=32):
    if cache_path and Path(cache_path).exists():
        print(f"Loading text feature cache: {cache_path}")
        cache = torch.load(cache_path, map_location='cpu', weights_only=False)
        print(f"   {len(cache)} samples loaded from cache")
        return cache

    print(f"Building text feature cache ({len(dataset)} samples)...")
    was_training = model.text_encoder.training
    model.text_encoder.eval()

    tmp_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    cache = {}
    for batch in tmp_loader:
        batch_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model.text_encoder(
            input_ids=batch_on_device['text_tokens'],
            attention_mask=batch_on_device['attention_mask']
        )
        raw_feats = outputs.last_hidden_state
        masks = batch_on_device['attention_mask']
        vids = batch['video_ids']
        for i in range(raw_feats.shape[0]):
            actual_len = int(masks[i].sum().item())
            cache[vids[i]] = raw_feats[i, :actual_len].cpu()

    if was_training:
        model.text_encoder.train()

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)
        size_mb = Path(cache_path).stat().st_size / 1024**2
        print(f"   Cache saved: {len(cache)} samples -> {size_mb:.1f} MB")
    return cache


# =============================================================================
# ROBUST SAVE
# =============================================================================
def _robust_save(save_dict: dict, target_path: Path, max_retries: int = 3):
    import shutil, tempfile, time
    target_path = Path(target_path)
    tmp_path = None
    for attempt in range(1, max_retries + 1):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt.tmp', dir=str(target_path.parent)) as tmp:
                tmp_path = Path(tmp.name)
            torch.save(save_dict, tmp_path)
            local_size = tmp_path.stat().st_size
            shutil.copy2(tmp_path, target_path)
            if local_size > 100 * 1024 * 1024:
                time.sleep(2)
            with open(target_path, 'r+b') as f:
                f.flush()
                os.fsync(f.fileno())
            os.sync()
            drive_size = target_path.stat().st_size
            if drive_size == local_size:
                if tmp_path and tmp_path.exists():
                    tmp_path.unlink()
                return True
        except Exception as e:
            print(f"    Save attempt {attempt} failed: {e}")
        finally:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
    torch.save(save_dict, target_path)
    return False


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    # Model Architecture (V11)
    'latent_dim': 256,
    'hidden_dim': 512,
    'num_fine_layers': 3,      # U-Net encoder/decoder layers
    'num_coarse_layers': 3,    # U-Net bottleneck layers
    'num_prior_layers': 4,
    'num_heads': 8,
    'dropout': 0.1,
    'max_seq_len': 512,
    'use_ssm_prior': True,
    'local_window': 64,        # Local attention window size

    # Training
    'batch_size': 32,
    'learning_rate': 1e-4,       # Higher for new modules (flow_block, length_predictor)
    'warmstart_lr_ratio': 0.05,  # Warm-started params get lr * 0.05 = 5e-6
    'length_lr_ratio': 1.0,     # Length predictor same as flow_block
    'warmup_epochs': 5,
    'num_epochs': 400,
    'gradient_clip': 1.0,
    'weight_decay': 0.02,

    # EMA
    'use_ema': True,
    'ema_decay': 0.999,

    # Loss Weights
    'W_FLOW': 1.0,
    'W_PRIOR': 0.5,            # Match V8 no_scst
    'W_LENGTH': 1.5,           # Higher: duration-aware predictor needs stronger signal
    'W_POSE': 5.0,             # Match V8 no_scst (best model) — strong pose accuracy
    'W_VELOCITY': 1.0,         # Match V8 no_scst — smooth motion
    'W_ACCEL': 0.0,            # Acceleration smoothness (2nd-order), 0 = off
    'W_SLT_PERCEPTUAL': 3,
    'W_SCST': 2.0,

    # SCST
    'use_scst': False,
    'scst_warmup_epochs': 10,
    'scst_steps': 20,
    'scst_cfg': 2.5,
    'slt_time_threshold': 0.3,  # Match V8 no_scst
    'scst_k': 4,
    'bleu_eval_samples': 200,

    # Sampling defaults for validation
    'val_ode_method': 'midpoint',
    'val_cfg_scale': 2.5,

    # Paths (UPDATE THESE)
    'stage1_checkpoint': 'checkpoints/17FEB/best_model.pt',
    'slt_model_dir': 'backTranslation_PHIX_model',
    'data_dir': 'SignFML_DataExtraction_Phoenix14T/data',
    'output_dir': 'checkpoints/v11_24mar',
    'warmstart_checkpoint': None,  # Set to None to start fresh or provide path via --warmstart_checkpoint

    'device': 'auto',
    'patience': 150,            # Match V8 no_scst
    'check_bleu_every': 1,
}


# =============================================================================
# PARTIAL WARM-START: Load V5/V10 weights into V11
# =============================================================================

def partial_warmstart(model_v11, v10_ckpt_path, device):
    """
    Load compatible weights from V10/V5 checkpoint into V11.

    Reuses: text_proj, ssm_prior, text_encoder (frozen)
    Skips: flow_block (new MultiScaleFlowBlock), length_predictor (new DurationAware)
    """
    if v10_ckpt_path is None or not os.path.exists(v10_ckpt_path):
        print(f"⏭️ Skipping warm-start: checkpoint {v10_ckpt_path} not found.")
        return None
    print(f"Loading V10 checkpoint for warm-start: {v10_ckpt_path}")
    ckpt = torch.load(v10_ckpt_path, map_location=device, weights_only=False)
    v10_sd = ckpt['model_state_dict']

    v11_sd = model_v11.state_dict()
    loaded_keys = []
    skipped_keys = []

    for key, param in v10_sd.items():
        if key in v11_sd and v11_sd[key].shape == param.shape:
            # Skip flow_block and length_predictor (new architecture)
            if 'flow_block' in key or 'length_predictor' in key:
                skipped_keys.append(key)
                continue
            v11_sd[key] = param
            loaded_keys.append(key)
        else:
            skipped_keys.append(key)

    model_v11.load_state_dict(v11_sd, strict=False)

    print(f"   Loaded {len(loaded_keys)} params (text_proj, ssm_prior, etc.)")
    print(f"   Skipped {len(skipped_keys)} params (flow_block, length_predictor = new arch)")

    # Show what was loaded
    loaded_modules = set(k.split('.')[0] for k in loaded_keys)
    print(f"   Loaded modules: {loaded_modules}")

    return ckpt


# =============================================================================
# SCST
# =============================================================================

def compute_scst_reward(hyps, refs):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import sacrebleu
    sf = SmoothingFunction().method1
    rewards = []
    for h, r in zip(hyps, refs):
        h_lower, r_lower = h.lower(), r.lower()
        h_tokens, r_tokens = h_lower.split(), r_lower.split()
        if not r_tokens:
            rewards.append(0.0)
            continue
        chrf_score = sacrebleu.sentence_chrf(h_lower, [r_lower]).score / 100.0
        b1 = sentence_bleu([r_tokens], h_tokens, weights=(1.0, 0, 0, 0), smoothing_function=sf)
        reward = max((chrf_score + b1) / 2.0, 0.01)
        rewards.append(reward)
    return torch.tensor(rewards)


@torch.no_grad()
def get_scst_samples(model, autoencoder, slt, batch, config, dataset_stats=None,
                     condition=None, text_mask=None, raw_text_feats=None):
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    lm = torch.as_tensor(config['latent_mean'], device=device).float()
    ls = torch.as_tensor(config['latent_std'], device=device).float()

    if dataset_stats is not None:
        mean_p = torch.as_tensor(dataset_stats['mean'], dtype=torch.float32, device=device)
        std_p = torch.as_tensor(dataset_stats['std'], dtype=torch.float32, device=device)

    model_raw = get_raw_model(model)

    # Greedy baseline
    latents_g, _, pred_len_g = model_raw.sample(
        batch, steps=config['scst_steps'], cfg_scale=config['scst_cfg'], temperature=0.0,
        condition=condition, text_mask=text_mask, raw_text_feats=raw_text_feats,
        return_length=True, ode_method='euler'  # Fast for SCST
    )
    latents_g_denorm = latents_g * ls + lm
    poses_g = autoencoder.decode(latents_g_denorm)
    if dataset_stats is not None:
        poses_g = poses_g * std_p + mean_p
    poses_g_list = [poses_g[i, :pred_len_g[i]] for i in range(poses_g.shape[0])]
    texts_g = slt.poses_to_text(poses_g_list)

    # Stochastic sample
    latents_s, x0_s, pred_len_s = model_raw.sample(
        batch, steps=config['scst_steps'], cfg_scale=config['scst_cfg'], temperature=1.0,
        condition=condition, text_mask=text_mask, raw_text_feats=raw_text_feats,
        return_length=True, ode_method='euler'
    )
    latents_s_denorm = latents_s * ls + lm
    poses_s = autoencoder.decode(latents_s_denorm)
    if dataset_stats is not None:
        poses_s = poses_s * std_p + mean_p
    poses_s_list = [poses_s[i, :pred_len_s[i]] for i in range(poses_s.shape[0])]
    texts_s = slt.poses_to_text(poses_s_list)

    reward_g = compute_scst_reward(texts_g, batch['texts']).to(device)
    reward_s = compute_scst_reward(texts_s, batch['texts']).to(device)

    if was_training:
        model.train()
    return latents_s, x0_s, reward_s, reward_g, pred_len_s


# =============================================================================
# SCHEDULER
# =============================================================================

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-7, offset_epoch=0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr'] / optimizer.param_groups[0].get('lr_ratio', 1.0)
        self.offset_epoch = offset_epoch

    def step(self, epoch):
        rel_epoch = max(0, epoch - self.offset_epoch)
        if rel_epoch < self.warmup_epochs:
            lr = 1e-8 + (self.base_lr - 1e-8) * (rel_epoch / max(1, self.warmup_epochs))
        else:
            progress = (rel_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2

        for param_group in self.optimizer.param_groups:
            ratio = param_group.get('lr_ratio', 1.0)
            param_group['lr'] = lr * ratio
        return lr


# =============================================================================
# TRAINING ENGINE
# =============================================================================

def train_epoch(model, autoencoder, loader, optimizer, scheduler, scaler, ema, slt, slt_perceptual, pose_loss_fn, device, config, epoch, dataset_stats=None):
    model.train()
    total_loss, total_fm_loss, num_batches = 0.0, 0.0, 0
    lm = torch.as_tensor(config['latent_mean'], device=device).float()
    ls = torch.as_tensor(config['latent_std'], device=device).float()

    for batch_idx, batch in enumerate(loader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        optimizer.zero_grad()

        # Encode poses to latent
        with torch.no_grad():
            gt_latent = autoencoder.encode(batch['poses'], mask=batch.get('pose_mask'))
            gt_latent = (gt_latent - lm) / ls

        results = model(batch, gt_latent=gt_latent)

        loss_main = config['W_FLOW'] * results['loss_flow'].mean() + \
                    config['W_PRIOR'] * results['loss_prior'].mean() + \
                    config['W_LENGTH'] * results['loss_length'].mean()

        # SCST
        loss_scst = torch.tensor(0.0, device=device)
        r_s_mean = r_g_mean = 0.0
        if config['use_scst'] and epoch >= config.get('scst_warmup_epochs', 0):
            model.eval()
            model_raw = get_raw_model(model)
            with torch.no_grad():
                latents_s, x0_s, r_s, r_g, len_s = get_scst_samples(
                    model, autoencoder, slt, batch, config, dataset_stats=dataset_stats
                )
            model.train()
            log_prob = model_raw.get_log_prob(latents_s, batch, x0=x0_s, pred_lengths=len_s, K=1)
            advantage = r_s - r_g
            loss_scst = -(advantage * log_prob).mean()
            r_s_mean = r_s.mean().item()
            r_g_mean = r_g.mean().item()

        # Pose + Perceptual + Velocity + Accel losses
        loss_perceptual = torch.tensor(0.0, device=device)
        loss_pose = torch.tensor(0.0, device=device)
        loss_velocity = torch.tensor(0.0, device=device)
        loss_accel = torch.tensor(0.0, device=device)

        need_pose_decode = (slt_perceptual is not None and config['W_SLT_PERCEPTUAL'] > 0) or \
                           (pose_loss_fn is not None and config.get('W_POSE', 0) > 0) or \
                           config.get('W_VELOCITY', 0) > 0 or \
                           config.get('W_ACCEL', 0) > 0
        if need_pose_decode:
            t = results['timesteps']
            est_x1 = results['est_x1']
            mask_slt = t > config['slt_time_threshold']

            if mask_slt.any():
                est_x1_sub = est_x1[mask_slt]
                gt_x1_sub = gt_latent[mask_slt]
                est_x1_denorm = est_x1_sub * ls + lm
                gt_x1_denorm = gt_x1_sub * ls + lm
                est_pose = autoencoder.decode(est_x1_denorm)
                with torch.no_grad():
                    gt_pose = autoencoder.decode(gt_x1_denorm)
                mask_sub = batch.get('pose_mask')[mask_slt] if 'pose_mask' in batch else None

                if slt_perceptual is not None and config['W_SLT_PERCEPTUAL'] > 0:
                    loss_perceptual = slt_perceptual(est_pose, gt_pose, mask=mask_sub)
                if pose_loss_fn is not None and config.get('W_POSE', 0) > 0:
                    loss_pose = pose_loss_fn(est_pose, gt_pose, mask=mask_sub)
                if config.get('W_VELOCITY', 0) > 0:
                    gt_vel = gt_pose[:, 1:, :] - gt_pose[:, :-1, :]
                    pred_vel = est_pose[:, 1:, :] - est_pose[:, :-1, :]
                    if mask_sub is not None:
                        vel_mask = (mask_sub[:, 1:] & mask_sub[:, :-1]).unsqueeze(-1)
                        vel_diff = F.smooth_l1_loss(pred_vel * vel_mask, gt_vel * vel_mask, reduction='sum', beta=0.1)
                        loss_velocity = vel_diff / (vel_mask.sum() * pred_vel.shape[-1]).clamp(min=1)
                    else:
                        loss_velocity = F.smooth_l1_loss(pred_vel, gt_vel, beta=0.1)
                    time_weight = t[mask_slt].clamp(min=0.2).mean().detach()
                    loss_velocity = loss_velocity * time_weight
                if config.get('W_ACCEL', 0) > 0:
                    gt_acc = gt_pose[:, 2:, :] - 2 * gt_pose[:, 1:-1, :] + gt_pose[:, :-2, :]
                    pred_acc = est_pose[:, 2:, :] - 2 * est_pose[:, 1:-1, :] + est_pose[:, :-2, :]
                    if mask_sub is not None:
                        acc_mask = (mask_sub[:, 2:] & mask_sub[:, 1:-1] & mask_sub[:, :-2]).unsqueeze(-1)
                        acc_diff = F.smooth_l1_loss(pred_acc * acc_mask, gt_acc * acc_mask, reduction='sum', beta=0.1)
                        loss_accel = acc_diff / (acc_mask.sum() * pred_acc.shape[-1]).clamp(min=1)
                    else:
                        loss_accel = F.smooth_l1_loss(pred_acc, gt_acc, beta=0.1)
                    loss_accel = loss_accel * time_weight

        total_batch_loss = loss_main + \
                           config['W_SCST'] * loss_scst.mean() + \
                           config['W_SLT_PERCEPTUAL'] * loss_perceptual.mean() + \
                           config.get('W_POSE', 0) * loss_pose.mean() + \
                           config.get('W_VELOCITY', 0) * loss_velocity.mean() + \
                           config.get('W_ACCEL', 0) * loss_accel.mean()

        if scaler:
            scaler.scale(total_batch_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(get_raw_model(model).parameters(), config['gradient_clip'])
            scaler.step(optimizer)
            scaler.update()
        else:
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(get_raw_model(model).parameters(), config['gradient_clip'])
            optimizer.step()

        if ema:
            ema.update()

        total_loss += total_batch_loss.item()
        total_fm_loss += loss_main.item()
        num_batches += 1

        if batch_idx % 20 == 0:
            scst_info = f", R_s={r_s_mean:.3f}, R_g={r_g_mean:.3f}" if config['use_scst'] else ""
            print(f"  Ep {epoch} [{batch_idx}/{len(loader)}]: Loss={total_batch_loss.item():.4f} "
                  f"(FM={loss_main.item():.4f}, Percep={loss_perceptual.item():.4f}, "
                  f"Pose={loss_pose.item():.4f}, Vel={loss_velocity.item():.4f}{scst_info})")

    return total_loss / max(num_batches, 1), total_fm_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, autoencoder, loader, slt, device, config, epoch, dataset_stats=None, ema=None, split_name='Val'):
    model.eval()
    all_hyps, all_refs = [], []
    lm = torch.as_tensor(config['latent_mean'], device=device).float()
    ls = torch.as_tensor(config['latent_std'], device=device).float()
    max_samples = config.get('bleu_eval_samples', 100)

    if ema is not None:
        ema.apply_shadow()

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Model BLEU (euler + 20 steps for speed)
        model_raw = get_raw_model(model)
        latents, _, seq_lens = model_raw.sample(
            batch, steps=20,
            cfg_scale=config.get('val_cfg_scale', 2.5),
            temperature=1.0, return_length=True,
            ode_method='euler',
            use_dynamic_cfg=True
        )
        poses = autoencoder.decode(latents * ls + lm)

        if dataset_stats is not None:
            mean_p = torch.as_tensor(dataset_stats['mean'], dtype=torch.float32, device=device)
            std_p = torch.as_tensor(dataset_stats['std'], dtype=torch.float32, device=device)
            poses = poses * std_p + mean_p

        poses_list = [poses[i, :seq_lens[i]] for i in range(poses.shape[0])]
        texts = slt.poses_to_text(poses_list)
        all_hyps.extend(texts)
        all_refs.extend(batch['texts'])

        if len(all_hyps) >= max_samples:
            break

    if ema is not None:
        ema.restore()

    import sacrebleu
    bleu4 = sacrebleu.corpus_bleu([h.lower() for h in all_hyps], [[r.lower() for r in all_refs]], tokenize="none").score
    ema_tag = " [EMA]" if ema is not None else ""
    print(f"  {split_name} BLEU-4: {bleu4:.2f} (n={len(all_hyps)}){ema_tag}")
    for i in range(min(3, len(all_hyps))):
        print(f"    REF:  {all_refs[i]}")
        print(f"    PRED: {all_hyps[i]}")
    return bleu4


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='SignFML V11 Training')
    parser.add_argument('--config', type=str, help='Config JSON')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--data-fraction', type=float, default=1.0)

    # Paths
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--stage1_checkpoint', type=str)
    parser.add_argument('--slt_model_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--warmstart_from', type=str, help='V10/V5 checkpoint for partial warm-start')
    parser.add_argument('--resume_from', type=str, help='V11 checkpoint to resume training')

    # Hyperparams
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--gradient_clip', type=float)
    parser.add_argument('--local_window', type=int, help='Local attention window size')

    # Loss weights
    parser.add_argument('--w_flow', type=float)
    parser.add_argument('--w_prior', type=float)
    parser.add_argument('--w_length', type=float)
    parser.add_argument('--w_pose', type=float)
    parser.add_argument('--w_velocity', type=float)
    parser.add_argument('--w_accel', type=float)
    parser.add_argument('--w_perceptual', type=float)
    parser.add_argument('--w_scst', type=float)
    parser.add_argument('--pred_length_mask_ratio', type=float, default=0.0)

    # SCST / Validation
    parser.add_argument('--use_scst', action='store_true')
    parser.add_argument('--patience', type=int)
    parser.add_argument('--check_bleu_every', type=int, default=5)
    parser.add_argument('--bleu_eval_samples', type=int)
    parser.add_argument('--val_ode_method', type=str, choices=['euler', 'midpoint', 'rk4'])
    parser.add_argument('--val_cfg_scale', type=float)
    parser.add_argument('--reset_best_val', action='store_true')
    parser.add_argument('--slt_on_cpu', action='store_true')
    parser.add_argument('--no_cache_text_features', action='store_true')
    parser.add_argument('--force_rebuild_cache', action='store_true')
    parser.add_argument('--text_cache_dir', type=str)

    # Ablation flags
    parser.add_argument('--no_prior', action='store_true', help='Ablation: disable coarse-to-fine prior')
    parser.add_argument('--no_slt_loss', action='store_true', help='Ablation: disable SLT Perceptual Loss')

    args = parser.parse_args()

    # Config
    config = DEFAULT_CONFIG.copy()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config.update(json.load(f))

    # CLI overrides
    if args.data_dir: config['data_dir'] = args.data_dir
    if args.stage1_checkpoint: config['stage1_checkpoint'] = args.stage1_checkpoint
    if args.output_dir: config['output_dir'] = args.output_dir
    if args.slt_model_dir: config['slt_model_dir'] = args.slt_model_dir
    if args.warmstart_from: config['warmstart_checkpoint'] = args.warmstart_from
    if args.resume_from: config['resume_checkpoint'] = args.resume_from
    if args.device: config['device'] = args.device
    if args.lr: config['learning_rate'] = args.lr
    if args.batch_size: config['batch_size'] = args.batch_size
    if args.epochs: config['num_epochs'] = args.epochs
    if args.gradient_clip: config['gradient_clip'] = args.gradient_clip
    if args.local_window: config['local_window'] = args.local_window
    if args.w_flow is not None: config['W_FLOW'] = args.w_flow
    if args.w_prior is not None: config['W_PRIOR'] = args.w_prior
    if args.w_length is not None: config['W_LENGTH'] = args.w_length
    if args.w_pose is not None: config['W_POSE'] = args.w_pose
    if args.w_velocity is not None: config['W_VELOCITY'] = args.w_velocity
    if args.w_accel is not None: config['W_ACCEL'] = args.w_accel
    if args.w_perceptual is not None: config['W_SLT_PERCEPTUAL'] = args.w_perceptual
    if args.w_scst is not None: config['W_SCST'] = args.w_scst
    if args.pred_length_mask_ratio is not None: config['pred_length_mask_ratio'] = args.pred_length_mask_ratio
    if args.patience: config['patience'] = args.patience
    if args.check_bleu_every: config['check_bleu_every'] = args.check_bleu_every
    if args.bleu_eval_samples: config['bleu_eval_samples'] = args.bleu_eval_samples
    if args.val_ode_method: config['val_ode_method'] = args.val_ode_method
    if args.val_cfg_scale: config['val_cfg_scale'] = args.val_cfg_scale
    if args.slt_on_cpu: config['slt_on_cpu'] = True
    if args.use_scst: config['use_scst'] = True
    config['cache_text_features'] = not args.no_cache_text_features
    if args.text_cache_dir: config['text_cache_dir'] = args.text_cache_dir

    # Ablation overrides
    if args.no_prior:
        config['use_ssm_prior'] = False
        config['W_PRIOR'] = 0.0
        print("⚠️  ABLATION: Coarse-to-fine prior DISABLED")
    if args.no_slt_loss:
        config['W_SLT_PERCEPTUAL'] = 0
        print("⚠️  ABLATION: SLT Perceptual Loss DISABLED")

    # Device
    if config['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['device'])
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load AE
    print("Loading Autoencoder...")
    ae = UnifiedPoseAutoencoder(pose_dim=534, latent_dim=256).to(device)
    ae_ckpt = torch.load(config['stage1_checkpoint'], map_location=device, weights_only=False)
    ae.load_state_dict(ae_ckpt['model_state_dict'] if 'model_state_dict' in ae_ckpt else ae_ckpt)
    ae.eval()
    ae.requires_grad_(False)

    # Load SLT
    print("Loading SLT...")
    s_device = 'cpu' if config.get('slt_on_cpu', False) else str(device)
    slt = SLTWrapper(model_dir=config['slt_model_dir'], device=s_device)

    # SLT Perceptual
    slt_perceptual = None
    if config.get('W_SLT_PERCEPTUAL', 0) > 0:
        print("Loading SLT Perceptual Loss...")
        slt_perceptual = SLTPerceptualLoss(slt_model_dir=config['slt_model_dir'], device=str(device))

    # Pose Loss
    pose_loss_fn = None
    if config.get('W_POSE', 0) > 0:
        pose_loss_fn = SignLanguageLoss(weight_body=0.10, weight_hands=0.45, weight_face=0.45)

    # Initialize V11 model
    print("Initializing LatentFlowMatcherV11...")
    model = LatentFlowMatcherV11(
        latent_dim=config['latent_dim'],
        hidden_dim=config['hidden_dim'],
        num_fine_layers=config['num_fine_layers'],
        num_coarse_layers=config['num_coarse_layers'],
        num_prior_layers=config['num_prior_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len'],
        use_ssm_prior=config['use_ssm_prior'],
        pred_length_mask_ratio=config.get('pred_length_mask_ratio', 0.0),
        local_window=config.get('local_window', 64)
    ).to(device)

    # Multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = nn.DataParallel(model)

    # ==========================================================================
    # WARM-START or RESUME
    # ==========================================================================
    ckpt = None
    start_epoch = 0
    best_bleu = -1.0
    best_loss = float('inf')
    patience_counter = 0

    if config.get('resume_checkpoint'):
        # Full resume from V11 checkpoint
        print(f"Resuming from V11 checkpoint: {config['resume_checkpoint']}")
        ckpt = torch.load(config['resume_checkpoint'], map_location=device, weights_only=False)
        msg = get_raw_model(model).load_state_dict(ckpt['model_state_dict'], strict=True)
        print(f"   Loaded V11 state (strict=True)")

        start_epoch = ckpt.get('epoch', -1) + 1
        if 'latent_mean' in ckpt:
            config['latent_mean'] = ckpt['latent_mean']
            config['latent_std'] = ckpt['latent_std']

    elif config.get('warmstart_checkpoint'):
        # Partial warm-start from V10/V5
        ckpt = partial_warmstart(get_raw_model(model), config['warmstart_checkpoint'], device)

        # Load latent stats from V10 checkpoint
        if 'latent_mean' in ckpt:
            config['latent_mean'] = ckpt['latent_mean']
            config['latent_std'] = ckpt['latent_std']
            print("   Loaded latent stats from V10 checkpoint.")
        elif 'config' in ckpt and 'latent_mean' in ckpt['config']:
            config['latent_mean'] = ckpt['config']['latent_mean']
            config['latent_std'] = ckpt['config']['latent_std']
            print("   Loaded latent stats from V10 config.")

        # Don't resume epoch/optimizer — train from epoch 0 with new params
        ckpt = None  # Prevent optimizer/EMA restore
        print("   Starting fresh training (epoch 0) with warm-started weights.")
    else:
        print("Training from scratch (no warm-start)")
        config['latent_mean'] = torch.zeros(256)
        config['latent_std'] = torch.ones(256)

    # ==========================================================================
    # OPTIMIZER: 3 LR groups
    # ==========================================================================
    model_raw = get_raw_model(model)

    # Categorize params
    warmstarted_params = []  # text_proj, ssm_prior
    new_flow_params = []     # flow_block (new architecture)
    new_length_params = []   # length_predictor (new architecture)

    for name, param in model_raw.named_parameters():
        if not param.requires_grad:
            continue
        if 'flow_block' in name:
            new_flow_params.append(param)
        elif 'length_predictor' in name:
            new_length_params.append(param)
        else:
            warmstarted_params.append(param)

    base_lr = config['learning_rate']
    warmstart_ratio = config.get('warmstart_lr_ratio', 0.05)

    param_groups = [
        {
            'params': warmstarted_params,
            'lr': base_lr * warmstart_ratio,
            'lr_ratio': warmstart_ratio,
            'name': 'warmstarted (text_proj, ssm_prior)'
        },
        {
            'params': new_flow_params,
            'lr': base_lr,
            'lr_ratio': 1.0,
            'name': 'flow_block (new)'
        },
        {
            'params': new_length_params,
            'lr': base_lr * config.get('length_lr_ratio', 1.0),
            'lr_ratio': config.get('length_lr_ratio', 1.0),
            'name': 'length_predictor (new)'
        }
    ]

    print(f"\nOptimizer LR groups:")
    for g in param_groups:
        n_params = sum(p.numel() for p in g['params'])
        print(f"  {g['name']}: lr={g['lr']:.2e}, params={n_params:,}")

    optimizer = AdamW(param_groups, weight_decay=config['weight_decay'])
    ema = EMA(model_raw, config['ema_decay']) if config['use_ema'] else None
    scaler = GradScaler() if device.type == 'cuda' else None

    scheduler = WarmupCosineScheduler(optimizer, config['warmup_epochs'], config['num_epochs'])

    # Restore optimizer/EMA from V11 resume
    if ckpt:
        if 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
                print("   Optimizer state restored.")
            except Exception as e:
                print(f"   Could not restore optimizer: {e}")

        if ema and 'ema_state_dict' in ckpt:
            try:
                ema.load_state_dict(ckpt['ema_state_dict'])
                for name in ema.shadow:
                    ema.shadow[name] = ema.shadow[name].to(device)
                print("   EMA state restored.")
            except Exception as e:
                print(f"   Could not restore EMA: {e}")

        if scaler and 'scaler_state_dict' in ckpt:
            try:
                scaler.load_state_dict(ckpt['scaler_state_dict'])
            except:
                pass

        for _key in ('val_bleu', 'best_bleu'):
            if _key in ckpt:
                best_bleu = ckpt[_key]
                if best_bleu <= 1.0:
                    best_bleu *= 100
                break
        best_loss = ckpt.get('best_loss', float('inf'))
        patience_counter = ckpt.get('patience_counter', 0)
        if not args.reset_best_val:
            print(f"   Best BLEU: {best_bleu:.2f}, Best Loss: {best_loss:.4f}")

    if args.reset_best_val:
        best_bleu = -1.0
        best_loss = float('inf')
        patience_counter = 0
        print("   Reset best values and patience.")

    # ==========================================================================
    # DATA
    # ==========================================================================
    dataset = SignLanguageDataset(data_path=config['data_dir'], split='train')
    base_stats = dataset.stats if hasattr(dataset, 'stats') else None

    if args.data_fraction < 1.0:
        indices = torch.randperm(len(dataset))[:int(len(dataset) * args.data_fraction)]
        dataset = torch.utils.data.Subset(dataset, indices)

    val_dataset = SignLanguageDataset(data_path=config['data_dir'], split='dev')
    test_dataset = SignLanguageDataset(data_path=config['data_dir'], split='test')

    # mBART cache
    actual_collate = collate_fn
    if config.get('cache_text_features', True):
        cache_dir = Path(config.get('text_cache_dir', config['output_dir'])) / 'text_cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        frac_tag = f"_f{args.data_fraction:.2f}" if args.data_fraction < 1.0 else ""

        if args.force_rebuild_cache:
            for p in cache_dir.glob("*.pt"):
                p.unlink()
                print(f"   Deleted old cache: {p}")

        train_cache = build_text_feature_cache(dataset, model_raw, device,
                                                cache_path=cache_dir / f"train{frac_tag}_mbart_feats.pt")
        val_cache = build_text_feature_cache(val_dataset, model_raw, device,
                                              cache_path=cache_dir / f"val{frac_tag}_mbart_feats.pt")
        test_cache = build_text_feature_cache(test_dataset, model_raw, device,
                                               cache_path=cache_dir / f"test{frac_tag}_mbart_feats.pt")

        dataset = CachedFeaturesDataset(dataset, train_cache)
        val_dataset = CachedFeaturesDataset(val_dataset, val_cache)
        test_dataset = CachedFeaturesDataset(test_dataset, test_cache)
        actual_collate = make_cached_collate_fn(collate_fn)

    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=actual_collate)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=actual_collate)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=actual_collate)

    # ==========================================================================
    # TRAINING LOOP
    # ==========================================================================
    out_dir = Path(config['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(out_dir / 'config.json', 'w') as f:
        json.dump({k: str(v) if isinstance(v, (torch.Tensor, Path)) else v
                    for k, v in config.items()}, f, indent=2)

    print(f"\nStarting V11 Training (epoch {start_epoch})...")
    print(f"  Output: {out_dir}")
    print(f"  Train: {len(dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    for epoch in range(start_epoch, config['num_epochs']):
        lr = scheduler.step(epoch)
        train_loss, train_fm_loss = train_epoch(
            model, ae, loader, optimizer, scheduler, scaler, ema, slt, slt_perceptual,
            pose_loss_fn, device, config, epoch, dataset_stats=base_stats
        )

        # Save best by FM loss
        if train_fm_loss < best_loss:
            best_loss = train_fm_loss
            save_dict = {
                'epoch': epoch, 'model_state_dict': model_raw.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config, 'latent_mean': config['latent_mean'],
                'latent_std': config['latent_std'], 'val_bleu': best_bleu,
                'best_loss': best_loss, 'patience_counter': patience_counter
            }
            if ema: save_dict['ema_state_dict'] = ema.state_dict()
            if scaler: save_dict['scaler_state_dict'] = scaler.state_dict()
            _robust_save(save_dict, out_dir / 'best_model.pt')
            print(f"  New best loss model (loss={best_loss:.4f})")

        # Periodic BLEU validation
        if epoch % config.get('check_bleu_every', 1) == 0:
            val_bleu = validate(model, ae, val_loader, slt, device, config, epoch,
                                dataset_stats=base_stats, ema=ema, split_name='Val')

            if val_bleu > best_bleu:
                best_bleu = val_bleu
                patience_counter = 0
                if ema is not None:
                    ema.apply_shadow()
                try:
                    save_dict = {
                        'epoch': epoch, 'model_state_dict': model_raw.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'config': config, 'latent_mean': config['latent_mean'],
                        'latent_std': config['latent_std'], 'val_bleu': val_bleu,
                        'best_loss': best_loss, 'patience_counter': patience_counter
                    }
                    if ema: save_dict['ema_state_dict'] = ema.state_dict()
                    if scaler: save_dict['scaler_state_dict'] = scaler.state_dict()
                    _robust_save(save_dict, out_dir / 'best_bleu_model.pt')
                    print(f"  New best BLEU model (BLEU={val_bleu:.2f}) [EMA]")
                finally:
                    if ema is not None:
                        ema.restore()
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{config.get('patience', 30)}")

            if patience_counter >= config.get('patience', 30):
                print(f"Early stopping at epoch {epoch}")
                break

        # Test BLEU disabled (save compute)
        # if epoch % config.get('check_bleu_every', 1) == 0 and patience_counter == 0:
        #     test_bleu = validate(model, ae, test_loader, slt, device, config, epoch,
        #                           dataset_stats=base_stats, ema=ema, split_name='Test')
        #     print(f"  Test BLEU-4: {test_bleu:.2f} [monitor]")

        bleu_str = f"{best_bleu:.2f}" if best_bleu >= 0 else "N/A"
        print(f"\nEpoch {epoch} | Loss: {train_loss:.4f}, FM: {train_fm_loss:.4f}, "
              f"Best_FM: {best_loss:.4f}, Best_BLEU: {bleu_str}, LR: {lr:.2e}")

        # Latest checkpoint
        latest_dict = {
            'epoch': epoch, 'model_state_dict': model_raw.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config, 'latent_mean': config['latent_mean'],
            'latent_std': config['latent_std'], 'val_bleu': best_bleu,
            'best_loss': best_loss, 'patience_counter': patience_counter
        }
        if ema: latest_dict['ema_state_dict'] = ema.state_dict()
        if scaler: latest_dict['scaler_state_dict'] = scaler.state_dict()
        _robust_save(latest_dict, out_dir / 'latest.pt')


if __name__ == '__main__':
    main()
