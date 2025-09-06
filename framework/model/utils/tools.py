from typing import List
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


def load_checkpoint(model, ckpt_path, *, eval_mode, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    logger.info("Model weights restored.")

    if eval_mode:
        model.eval()
        logger.info("Model in eval mode.")


def detach_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


# def create_one_hot(keyids: List[str], IDs_list: List, IDs_labels: List[List], one_hot_dim: List):
#     style_batch = []
#     for keyid in keyids:
#         keyid = keyid.split('_')
#         identity_vector = IDs_labels[IDs_list.index(keyid[0])]
#         emotion_idx = int(keyid[2])
#         emotion_vector = torch.eye(one_hot_dim[1])[emotion_idx]
#         intensity_idx = int(keyid[3])
#         intensity_vector = torch.eye(one_hot_dim[2])[intensity_idx]
#         style_vector = torch.cat([identity_vector, emotion_vector, intensity_vector], dim=0)
#         style_batch.append(style_vector)
#     style_batch = torch.stack(style_batch, dim=0)
#     return style_batch

# emotion and intensity comes from audio_exraction not the label 
# get emotion and intensity dims, and add it to the audio feature dim
import torch
import numpy as np
from typing import List, Dict

def get_shape_para(keyids: List[str], identity_dict: Dict[str, np.ndarray]):
    shape_batch = []
    for keyid in keyids:
        # 根据下划线分割后，取第一个部分作为身份标识
        identity = keyid.split('_')[0]
        # 若身份不存在则抛出错误（或根据需求进行处理）
        if identity not in identity_dict:
            raise ValueError(f"身份 {identity} 在数据字典中未找到！")
        # 获取对应的 array 数据，并转换为 tensor
        shape_vec = identity_dict[identity]
        if isinstance(shape_vec, np.ndarray):
            shape_vec = torch.from_numpy(shape_vec)
        shape_batch.append(shape_vec)
    # 堆叠所有 tensor
    shape_batch = torch.stack(shape_batch, dim=0)
    return shape_batch


def create_one_hot(keyids: List[str], IDs_list: List, IDs_labels: List[List], one_hot_dim: List):
    style_batch = []
    for keyid in keyids:
        keyid = keyid.split('_')
        identity_vector = IDs_labels[IDs_list.index(keyid[0])]
        emotion_idx = int(keyid[2])
        emotion_vector = torch.eye(one_hot_dim[1])[emotion_idx]
        intensity_idx = int(keyid[3])
        intensity_vector = torch.eye(one_hot_dim[2])[intensity_idx]
        # print("****************************\n\t style vector shape:\n******************************",identity_vector.shape, emotion_vector.shape, intensity_vector.shape)
        # torch.Size([32]) torch.Size([8]) torch.Size([3])
        style_vector = torch.cat([identity_vector, emotion_vector, intensity_vector], dim=0)
        style_batch.append(style_vector)
    style_batch = torch.stack(style_batch, dim=0)
    return style_batch


def resample_input(audio_embed, motion_embed, ifps, ofps, emotion_embed=None):
    if ifps % ofps == 0:
        factor = -1 * (-ifps // ofps)

        # Make audio length even using interpolation if needed
        if audio_embed.shape[1] % 2 != 0:
            target_len = audio_embed.shape[1] - 1
            audio_embed = F.interpolate(audio_embed.permute(0, 2, 1), size=target_len, mode='linear', align_corners=True).permute(0, 2, 1)
            if emotion_embed is not None:
                emotion_embed = F.interpolate(emotion_embed.permute(0, 2, 1), size=target_len, mode='linear', align_corners=True).permute(0, 2, 1)

        # Match audio length to 2x motion (or vice versa)
        expected_audio_len = motion_embed.shape[1] * 2
        if audio_embed.shape[1] > expected_audio_len:
            audio_embed = F.interpolate(audio_embed.permute(0, 2, 1), size=expected_audio_len, mode='linear', align_corners=True).permute(0, 2, 1)
            if emotion_embed is not None:
                emotion_embed = F.interpolate(emotion_embed.permute(0, 2, 1), size=expected_audio_len, mode='linear', align_corners=True).permute(0, 2, 1)
        elif audio_embed.shape[1] < expected_audio_len:
            motion_len = audio_embed.shape[1] // 2
            motion_embed = F.interpolate(motion_embed.permute(0, 2, 1), size=motion_len, mode='linear', align_corners=True).permute(0, 2, 1)

    else:
        factor = -1 * (-ifps // ofps)
        target_len = motion_embed.shape[1] * factor

        # Audio
        audio_embed = F.interpolate(audio_embed.permute(0, 2, 1), size=target_len, mode='linear', align_corners=True).permute(0, 2, 1)

        # Emotion
        if emotion_embed is not None:
            emotion_embed = F.interpolate(emotion_embed.permute(0, 2, 1), size=target_len, mode='linear', align_corners=True).permute(0, 2, 1)

    # Reshape both
    batch_size = motion_embed.shape[0]
    audio_embed = torch.reshape(audio_embed, (batch_size, audio_embed.shape[1] // factor, audio_embed.shape[2] * factor))

    if emotion_embed is not None:
        emotion_embed = torch.reshape(emotion_embed, (batch_size, emotion_embed.shape[1] // factor, emotion_embed.shape[2] * factor))
        return audio_embed, motion_embed, emotion_embed
    else:
        return audio_embed, motion_embed
