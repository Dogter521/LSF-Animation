import torch
import torch.nn.functional as F
from torch.nn import Module
from hydra.utils import instantiate
from pathlib import Path
import logging
import pickle
from deps.flame.flame_pytorch import FLAME

logger = logging.getLogger(__name__)


class ComputeLosses(Module):
    """Loss computation for VqvaePredict (FDD setting).

    * Eliminates all NumPy ops — completely differentiable in PyTorch.
    * Supports **StandardScaler** and **MinMaxScaler** pickles for expression & jaw.
    * Keeps running averages for easy Lightning logging via ``compute()``.
    """

    # ------------------------------------------------------------
    def __init__(
        self,
        *,
        region_path: str = "/data/lx22/manba/face/work_25/ProbTalk3D_iden_split/datasets/regions",
        λ_vel: float = 0.1,
        λ_var: float = 0.1,
        λ_fft: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        # -------- 1. running-sum buffers --------
        self.loss_names = [
            "latent_manifold",
            "recons_exp",
            "recons_jaw",
            "vel",
            "var",
            "fft",
            "total",
        ]
        for ln in self.loss_names + ["count"]:
            self.register_buffer(ln, torch.tensor(0.0))

        # -------- 2. atomic losses & λ --------
        self._losses_func = {
            ln: instantiate(kwargs[ln + "_func"])
            for ln in ["latent_manifold", "recons_exp", "recons_jaw"]
        }
        self._params = {
            "latent_manifold": kwargs.get("latent_manifold", 1.0),
            "recons_exp": kwargs.get("recons_exp", 1.0),
            "recons_jaw": kwargs.get("recons_jaw", 1.0),
            "vel": λ_vel,
            "var": λ_var,
            "fft": λ_fft,
        }

        # -------- 3. FLAME --------
        flame_cfg = {
            "flame_model_path": "/data/lx22/manba/face/work_25/ProbTalk3D_iden_split/deps/flame/model/generic_model.pkl",
            "static_landmark_embedding_path": "/data/lx22/manba/face/work_25/ProbTalk3D_iden_split/deps/flame/model/flame_static_embedding.pkl",
            "dynamic_landmark_embedding_path": "/data/lx22/manba/face/work_25/ProbTalk3D_iden_split/deps/flame/model/flame_dynamic_embedding.npy",
            "shape_params": 300,
            "expression_params": 100,
            "pose_params": 6,
            "use_face_contour": True,
            "use_3D_translation": True,
            "optimize_eyeballpose": True,
            "optimize_neckpose": True,
            "batch_size": 1,
        }
        from types import SimpleNamespace

        self.flame_layer = FLAME(SimpleNamespace(**flame_cfg))
        self.flame_layer.eval()
        for param in self.flame_layer.parameters():
            param.requires_grad = False

        # -------- 4. load scalers (torch constants) --------
        exp_scaler = pickle.load(
            open(
                "/data/lx22/manba/face/work_25/ProbTalk3D_iden_split/datasets/scaler_exp.pkl",
                "rb",
            )
        )
        jaw_scaler = pickle.load(
            open(
                "/data/lx22/manba/face/work_25/ProbTalk3D_iden_split/datasets/scaler_jaw.pkl",
                "rb",
            )
        )

        def _extract_affine(scaler, dim_slice):
            """Return (offset, scale) so that *inverse* is ``x * scale + offset``."""
            if hasattr(scaler, "mean_"):
                # StandardScaler
                offset = torch.tensor(scaler.mean_[dim_slice], dtype=torch.float32)
                scale = torch.tensor(scaler.scale_[dim_slice], dtype=torch.float32)
            elif hasattr(scaler, "data_min_"):
                # MinMaxScaler
                offset = torch.tensor(scaler.data_min_[dim_slice], dtype=torch.float32)
                scale = torch.tensor(scaler.scale_[dim_slice], dtype=torch.float32)
            else:
                raise AttributeError("Unsupported scaler type (expect Standard/MinMax).")
            return offset, scale

        exp_off, exp_scale = _extract_affine(exp_scaler, slice(0, 50))
        jaw_off, jaw_scale = _extract_affine(jaw_scaler, slice(None))
        self.register_buffer("exp_offset", exp_off)
        self.register_buffer("exp_scale", exp_scale)
        self.register_buffer("jaw_offset", jaw_off)
        self.register_buffer("jaw_scale", jaw_scale)

        # -------- 5. upper‑face vertex mask --------
        with open(Path(region_path) / "fdd.txt") as f:
            idx = [int(i) for i in f.read().strip().split(",")]
        mask = (torch.tensor(idx).unsqueeze(-1) * 3 + torch.arange(3)).flatten()
        self.register_buffer("upper_mask", mask)

    # ============================================================
    # helpers
    # ============================================================
    @staticmethod
    def _diff(x: torch.Tensor, order: int = 1):
        return x if order == 0 else ComputeLosses._diff(x[..., 1:, :] - x[..., :-1, :], order - 1)

    @staticmethod
    def _log_psd(x: torch.Tensor, eps: float = 1e-6):
        fft = torch.fft.rfft(x, dim=1)
        psd = (fft.real**2 + fft.imag**2).mean(-1)
        return torch.log(psd + eps)

    # ---- Param → verts ----
    def _inverse_affine(self, x: torch.Tensor, offset: torch.Tensor, scale: torch.Tensor):
        return x * scale + offset

    def _exp_jaw_to_verts(self, exp100: torch.Tensor, jaw3: torch.Tensor):
        """
        Args:
            exp100: (B, T, 100)
            jaw3: (B, T, 3)
        Returns:
            verts: (B, T, 5023, 3)
        """
        B, T, _ = exp100.shape
        dev = exp100.device
        BsT = B * T

        # 补齐 pose: global pose 全0 + jaw3
        input_global_pose = torch.zeros(BsT, 3, device=dev)
        input_jaw_pose = jaw3.reshape(BsT, 3)
        input_pose = torch.cat([input_global_pose, input_jaw_pose], dim=1)  # [BsT, 6]
        
        

        verts, _ = self.flame_layer(
            torch.zeros(BsT, 300, device=dev),     # shape: 全0
            exp100.reshape(BsT, 100),              # exp: [BsT, 100]
            input_pose.float()                     # pose: [BsT, 6]
        )
        return verts.view(B, T, 5023, 3)


    # ---- geometry losses ----
    def _velocity_loss(self, vp, vg, mask):
        vp = vp.reshape(*vp.shape[:2], -1)[..., mask]
        vg = vg.reshape(*vg.shape[:2], -1)[..., mask]
        return F.l1_loss(self._diff(vp), self._diff(vg))

    def _variance_loss(self, vp, vg, mask):
        vp = vp.reshape(*vp.shape[:2], -1)[..., mask]
        vg = vg.reshape(*vg.shape[:2], -1)[..., mask]
        return F.l1_loss(vp.var(dim=1), vg.var(dim=1))

    def _spectral_loss(self, vp, vg, mask):
        vp = vp.reshape(*vp.shape[:2], -1)[..., mask]
        vg = vg.reshape(*vg.shape[:2], -1)[..., mask]
        return F.l1_loss(self._log_psd(vp), self._log_psd(vg))

    # ============================================================
    # main entry
    # ============================================================
    def update(
        self,
        *,
        motion_quant_pred=None,
        motion_quant_ref=None,
        motion_pred=None,
        motion_ref=None,
    ):
        dev = motion_pred.device
        total = torch.tensor(0.0, device=dev)

        # base losses
        total += self._update_loss("latent_manifold", outputs=motion_quant_pred, inputs=motion_quant_ref)
        total += self._update_loss("recons_exp", outputs=motion_pred[..., :50], inputs=motion_ref[..., :50])
        total += self._update_loss("recons_jaw", outputs=motion_pred[..., 50:], inputs=motion_ref[..., 50:])


        # GT verts
        exp50_gt = motion_ref[:, :, :50]
        jaw3_gt = motion_ref[:, :, 50:]
        # B, T, _ = exp50_gt.shape
        # dev = exp50_gt.device
        # BsT = B * T
        # flame_cfg = self.flame_cfg
        # flame_cfg.batch_size = BsT
        # flame_layer = FLAME(flame_cfg).to(dev)

        # logger.info("exp50_gt.shape: %s, jaw3_gt.shape: %s", exp50_gt.shape, jaw3_gt.shape)


        verts_g = self._exp_jaw_to_verts(torch.cat([exp50_gt, torch.zeros_like(exp50_gt)], -1), jaw3_gt)

        # Pred verts (affine inverse)
        exp50_inv = self._inverse_affine(motion_pred[:,:, :50], self.exp_offset, self.exp_scale)
        jaw3_inv = self._inverse_affine(motion_pred[:,:, 50:], self.jaw_offset, self.jaw_scale)
        verts_p = self._exp_jaw_to_verts(torch.cat([exp50_inv, torch.zeros_like(exp50_inv)], -1), jaw3_inv)

        # vel/var/fft
        mask = self.upper_mask
        vel = self._velocity_loss(verts_p, verts_g, mask)
        var = self._variance_loss(verts_p, verts_g, mask)
        fft = self._spectral_loss(verts_p, verts_g, mask)
        # "vel", "var", "fft"
        # for name, val in zip(["vel", "var", "fft"], [vel, var, fft]):
        #     getattr(self, name).add_(val.detach())
        #     total += self._params[name] * val
        name = "var"
        val = var  # Only add vel loss
        getattr(self, name).add_(val.detach())
        total += self._params[name] * val


        # ---------- (4) 累计总损 ----------
        self.total += total.detach()
        self.count += 1
        return total

    # ============================================================
    # util
    # ============================================================
    def compute(self):
        count = self.count
        loss_dict = {loss: getattr(self, loss) / count for loss in self.loss_names}
        return loss_dict

    def _update_loss(self, loss: str, *, outputs=None, inputs=None):
        # Update the loss
        val = self._losses_func[loss](outputs, inputs)
        getattr(self, loss).__iadd__(val.detach())
        # Return a weighted sum
        weighted_loss = self._params[loss] * val
        return weighted_loss

    def loss_to_logname(self, loss: str, split: str):
        if "_" in loss:
            loss_type, name = loss.split("_", 1)
            log_name = f"{loss_type}/{name}/{split}"
        else:
            log_name = f"{loss}/{split}"
        return log_name
