import os
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import logging
from types import SimpleNamespace

from deps.flame.flame_pytorch import FLAME
from framework.model.utils.tools import detach_to_numpy

# ========== CLI 参数 ==========
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True, help="Base path to results/evaluation/... directory (e.g., 'results/evaluation/vae_pred/1/20_multi')")
args = parser.parse_args()

# ========== 路径设置 ==========
base_path = Path(args.path)
param_dir = base_path 
vertex_dir = base_path / "vertex"
gt_param_dir = Path("/data/lx22/manba/face/work_25/ProbTalk3D_iden_split/datasets/mead/param")

vertex_dir.mkdir(parents=True, exist_ok=True)

# ========== 日志设置 ==========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== 构建 FLAME 模型 ==========
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
flame = FLAME(SimpleNamespace(**flame_cfg)).cuda().eval()

# ========== 转换函数 ==========
def transfer_to_vert(flamelayer, shape, exp, jaw, save_path, device, idx, i):
    input_shape = torch.tensor(shape).to(device)
    input_exp = torch.tensor(exp).to(device)
    input_global_pose = torch.zeros(jaw.shape[0], 3).to(device)
    input_jaw_pose = torch.tensor(jaw).to(device)
    input_pose = torch.cat((input_global_pose, input_jaw_pose), dim=1)
    vertices, _ = flamelayer(input_shape.float(), input_exp.float(), input_pose.float())
    vertices_npy = detach_to_numpy(vertices)
    if (idx + 1) % 100 == 0 and (i + 1) % 3 == 1:
        logger.info(f"Saving vert: {save_path.stem}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, vertices_npy)
    return vertices_npy

# ========== 批量转换 ==========
param_files = sorted(param_dir.glob("*.npy"))

for idx, param_path in enumerate(tqdm(param_files, desc="Converting params to verts")):
    filename = param_path.stem  # e.g. M003_002_0_0_disgusted_low_0
    parts = filename.split("_")
    # if len(parts) < 7:
    #     logger.warning(f"Skipping invalid file: {filename}")
    #     continue

    keyid = "_".join(parts[:4])  # e.g. M003_002_0_0
    suffix = parts[-1]           # could be 0~9, 'gt', or 'one'

    # 获取 sample index
    if suffix in ["gt", "one"]:
        sample_idx = 0
    else:
        try:
            sample_idx = int(suffix)
        except ValueError:
            logger.warning(f"Skipping unknown suffix file: {filename}")
            continue

    # 获取 shape 参数（来自 GT）
    gt_path = gt_param_dir / f"{keyid}.npy"
    if not gt_path.exists():
        logger.warning(f"Missing GT file: {gt_path}")
        continue
    gt_data = np.load(gt_path)  # (1, T, 403)
    shape_first = gt_data[0, 0, :300]

    # 加载预测参数
    pred_data = np.load(param_path)[0]  # (T, 403)
    T = pred_data.shape[0]
    shape_seq = np.tile(shape_first, (T, 1))  # (T, 300)
    exp_seq = pred_data[:, 300:400]
    jaw_seq = pred_data[:, 400:]

    # 保存
    save_path = vertex_dir / f"{filename}.npy"
    transfer_to_vert(
        flamelayer=flame,
        shape=shape_seq,
        exp=exp_seq,
        jaw=jaw_seq,
        save_path=save_path,
        device="cuda",
        idx=idx,
        i=sample_idx
    )
