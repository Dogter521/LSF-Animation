import os
import logging
from typing import List

import hydra
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from hydra.utils import instantiate
from torch import Tensor
from torch.nn.functional import mse_loss

from deps.flame.flame_pytorch import FLAME
from framework.data.utils import get_split_keyids
from framework.model.base import BaseModel
from framework.model.feature_extractor.emotion2vec.scripts import extract_features
from framework.model.metrics.compute import ComputeMetrics
from framework.model.utils.tools import load_checkpoint, resample_input

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# FLAME config (explicit)
# -----------------------------------------------------------------------------
FLAME_CFG = {
    "flame_model_path": "deps/flame/model/generic_model.pkl",
    "static_landmark_embedding_path": "deps/flame/model/flame_static_embedding.pkl",
    "dynamic_landmark_embedding_path": "deps/flame/model/flame_dynamic_embedding.npy",
    "shape_params": 300,
    "expression_params": 100,
    "pose_params": 6,
    "use_face_contour": True,
    "use_3D_translation": True,
    "optimize_eyeballpose": True,
    "optimize_neckpose": True,
    "num_worker": 4,
    "batch_size": 32,
    "ring_margin": 0.5,
    "ring_loss_weight": 1.0,
}

# -----------------------------------------------------------------------------
# Helpers for FDD‑oriented losses (velocity / variance / spectral)
# -----------------------------------------------------------------------------

def _diff(x: torch.Tensor, order: int = 1):
    """Temporal finite difference of arbitrary order."""
    if order == 0:
        return x
    d1 = x[..., 1:, :] - x[..., :-1, :]
    return _diff(d1, order - 1)


def _batch_psd(x: torch.Tensor, eps: float = 1e-6):
    """Return log‑PSD averaged over feature dimension."""
    fft = torch.fft.rfft(x, dim=1)
    psd = (fft.real ** 2 + fft.imag ** 2).mean(-1)
    return torch.log(psd + eps)


def fdd_velocity_loss(v_pred, v_gt, upper_mask, order: int = 1):
    v_pred = v_pred.view(*v_pred.shape[:2], -1)[..., upper_mask]
    v_gt = v_gt.view(*v_gt.shape[:2], -1)[..., upper_mask]
    return F.l1_loss(_diff(v_pred, order), _diff(v_gt, order))


def fdd_variance_loss(v_pred, v_gt, upper_mask):
    v_pred = v_pred.view(*v_pred.shape[:2], -1)[..., upper_mask]
    v_gt = v_gt.view(*v_gt.shape[:2], -1)[..., upper_mask]
    return F.l1_loss(v_pred.var(dim=1), v_gt.var(dim=1))


def fdd_spectral_loss(v_pred, v_gt, upper_mask):
    v_pred = v_pred.view(*v_pred.shape[:2], -1)[..., upper_mask]
    v_gt = v_gt.view(*v_gt.shape[:2], -1)[..., upper_mask]
    return F.l1_loss(_batch_psd(v_pred), _batch_psd(v_gt))

# -----------------------------------------------------------------------------
# VqvaePredict with explicit FLAME & FDD‑losses
# -----------------------------------------------------------------------------

class VqvaePredict(BaseModel):
    def __init__(
        self,
        nfeats: int,
        split_path: str,
        one_hot_dim: List,
        resumed_training: bool,
        λ_vel: float = 1.0,
        λ_var: float = 0.3,
        λ_fft: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # ----- FDD & FLAME resources -----
        upper_map_path = "datasets/regions/fdd.txt"
        upper_map_np = np.loadtxt(upper_map_path, delimiter=",", dtype=int)
        self.register_buffer("upper_mask", torch.tensor(upper_map_np).unsqueeze(-1) * 3 + torch.arange(3))
        self.upper_mask = self.upper_mask.flatten()  # [1501*3]

        with open("datasets/scaler_exp.pkl", "rb") as f:
            self.scaler_exp = pickle.load(f)
        with open("datasets/scaler_jaw.pkl", "rb") as f:
            self.scaler_jaw = pickle.load(f)

        self.flame = FLAME(FLAME_CFG)

        # ----- original init (truncated to essentials) -----
        self.feature_extractor = instantiate(self.hparams.feature_extractor)
        self.audio_encoded_dim = self.feature_extractor.audio_encoded_dim
        self.all_identity_list = get_split_keyids(path=split_path, split="train")
        self.identity_extractor = nn.Sequential(
            nn.Linear(300, 256), nn.ReLU(), nn.Linear(256, len(self.all_identity_list)), nn.Softmax(-1)
        )

        # ... (omitted: motion_prior, feature_predictor, etc.) ...

        self.metrics = ComputeMetrics()

    # ------------------------------------------------------------------
    # Param → Vertices (vectorised, differentiable)
    # ------------------------------------------------------------------
    def _params_to_vertices(self, motion: torch.Tensor):
        """motion: [B, T, 103] (50 exp + 3 jaw) | returns [B, T, 5023, 3]."""
        B, T, _ = motion.shape
        device = motion.device
        shape = torch.zeros(B, T, 300, device=device)
        exp50 = motion[..., :50]
        jaw3 = motion[..., 50:]
        exp100 = torch.cat([exp50, torch.zeros_like(exp50)], dim=-1)

        # inverse scaling to real exp / jaw values
        BsT = B * T
        exp_np = self.scaler_exp.inverse_transform(exp100.reshape(BsT, -1).cpu()).astype(np.float32)
        jaw_np = self.scaler_jaw.inverse_transform(jaw3.reshape(BsT, -1).cpu()).astype(np.float32)
        exp100 = torch.from_numpy(exp_np).to(device).view(B, T, -1)
        jaw3 = torch.from_numpy(jaw_np).to(device).view(B, T, -1)

        global_pose = torch.zeros(B, T, 3, device=device)
        pose = torch.cat([global_pose, jaw3], dim=-1)

        verts, _ = self.flame(
            shape.view(BsT, -1).float(), exp100.view(BsT, -1).float(), pose.view(BsT, -1).float()
        )
        return verts.view(B, T, 5023, 3)

    # ------------------------------------------------------------------
    # Training step with FDD losses integrated
    # ------------------------------------------------------------------
    def allsplit_step(self, split: str, batch, batch_idx):
        # ... (audio & identity code unchanged) ...
        prediction = self.feature_predictor(batch["audio"], emotion_features, style_one_hot)
        motion_quant_pred, _, _ = self.motion_prior.quantize(prediction)
        motion_pred = self.motion_prior.motion_decoder(motion_quant_pred)  # [B, T, 53]

        motion_ref = batch["motion"]
        loss_rec = mse_loss(motion_pred, motion_ref)

        # --- FDD extras (on vertices) ---
        verts_pred = self._params_to_vertices(motion_pred[..., :53])
        verts_gt = self._params_to_vertices(motion_ref[..., :53]).detach()

        loss_vel = fdd_velocity_loss(verts_pred, verts_gt, self.upper_mask)
        loss_var = fdd_variance_loss(verts_pred, verts_gt, self.upper_mask)
        loss_fft = fdd_spectral_loss(verts_pred, verts_gt, self.upper_mask)

        loss_total = (
            loss_rec
            + self.hparams.λ_vel * loss_vel
            + self.hparams.λ_var * loss_var
            + self.hparams.λ_fft * loss_fft
        )

        self.log_dict(
            {
                f"{split}/loss": loss_total,
                f"{split}/rec": loss_rec,
                f"{split}/vel": loss_vel,
                f"{split}/var": loss_var,
                f"{split}/fft": loss_fft,
            },
            prog_bar=True,
        )

        if split == "val":
            self.metrics.update(motion_pred.detach(), motion_ref.detach(), [motion_ref.shape[1]] * motion_ref.shape[0])

        return loss_total
