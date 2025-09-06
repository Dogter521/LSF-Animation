import logging
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from rich.progress import track

from .base import BaseDataModule
from .utils import get_split_keyids
from framework.data.tools.collate import audio_normalize
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class MeadDataModule(BaseDataModule):
    def __init__(self, batch_size: int,
                 num_workers: int,
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         num_workers=num_workers)
        self.save_hyperparameters(logger=False)

        self.split_path = self.hparams.split_path
        self.one_hot_dim = self.hparams.one_hot_dim
        self.Dataset = MEAD
        # Get additional info of the dataset
        sample_overrides = {"split": "train", "tiny": True,
                            "progress_bar": False}
        self._sample_set = self.get_sample_set(overrides=sample_overrides)
        self.nfeats = self._sample_set.nfeats


class MEAD(Dataset):
    def __init__(self, data_name: str,
                 motion_path: str,
                 audio_path: str,
                 split_path: str,
                 load_audio: str,
                 split: str,
                 tiny: bool,
                 progress_bar: bool,
                 debug: bool,
                 **kwargs):
        super().__init__()
        self.data_name = data_name
        self.load_audio = load_audio

        ids = get_split_keyids(path=split_path, split=split)
        if progress_bar:
            enumerator = enumerate(track(ids, f"Loading {data_name} {split} dataset"))
        else:
            enumerator = enumerate(ids)

        if tiny:
            max_data = 2
        elif not tiny and debug:
            max_data = 8
        else:
            max_data = np.inf

        motion_data_all = {}
        shape_data_all = {}
        audio_data_all = {}

        # Initializing MinMaxScalars for exp and jaw
        exp_scaler = MinMaxScaler()
        exp_sample = [-3, 3]        # accepted exp value range
        a = np.array(exp_sample)[:, np.newaxis]
        a = np.repeat(a, 50, axis=1)
        exp_scaler.fit(a)           # scales data to [0,1] range

        jaw_scaler = MinMaxScaler()
        jaw_sample = [-0.1, 0.5]    # accepted jaw values range
        b = np.array(jaw_sample)[:, np.newaxis]
        b = np.repeat(b, 3, axis=1)
        jaw_scaler.fit(b)           # scales data to [0,1] range

        if load_audio:
            for i, id in enumerator:
                if len(motion_data_all) >= max_data:
                    break
                # load 3DMEAD dataset
                key, motion_data, shape_data, audio_data = load_data(keyid=id,
                                                                     motion_path=Path(motion_path),
                                                                     audio_path=Path(audio_path),
                                                                     max_data=max_data,
                                                                     load_audio=load_audio,
                                                                     exp_scaler=exp_scaler,
                                                                     jaw_scaler=jaw_scaler,
                                                                     split=split)
                motion_data_all.update(dict(zip(key, motion_data)))
                shape_data_all.update(dict(zip(key, shape_data)))
                audio_data_all.update(dict(zip(key, audio_data)))
        else:
            for i, id in enumerator:
                if len(motion_data_all) >= max_data:
                    break
                # load 3DMEAD dataset
                key, motion_data, shape_data = load_data(keyid=id,
                                                         motion_path=Path(motion_path),
                                                         audio_path=Path(audio_path),
                                                         max_data=max_data,
                                                         load_audio=load_audio,
                                                         exp_scaler=exp_scaler,
                                                         jaw_scaler=jaw_scaler,
                                                         split=None)
                motion_data_all.update(dict(zip(key, motion_data)))
                shape_data_all.update(dict(zip(key, shape_data)))

        self.motion_data = motion_data_all
        self.shape_data = shape_data_all
        if load_audio:
            self.audio_data = audio_data_all
        self.keyids = list(motion_data_all.keys())  # file name
        self.nfeats = self[0]["motion"].shape[2]    # number of feature
        print(f"The number of loaded data pair is: {len(self.motion_data)}")
        print(f"Number of features of a motion frame: {self.nfeats}")

    def load_keyid(self, keyid):
        if self.load_audio:
            element = {"motion": self.motion_data[keyid], "shape": self.shape_data[keyid], "audio": self.audio_data[keyid],
                       "keyid": keyid}
        else:
            element = {"motion": self.motion_data[keyid], "shape": self.shape_data[keyid], "keyid": keyid}
        return element

    def __getitem__(self, index):
        keyid = self.keyids[index]
        element = self.load_keyid(keyid)
        return element

    def __len__(self):
        return len(self.keyids)

    def __repr__(self):
        return f"{self.data_name} dataset: ({len(self)}, _, ..)"


def load_data(keyid, motion_path, audio_path, max_data, load_audio, exp_scaler, jaw_scaler, split):
    try:
        motion_dir = list(motion_path.glob(f"{keyid}*.npy"))
        motion_key = [directory.stem for directory in motion_dir]
        audio_dir = list(audio_path.glob(f"{keyid}*.wav"))
        audio_key = [directory.stem for directory in audio_dir]
    except FileNotFoundError:
        return None

    keys = []
    motion_data = []
    shape_data = []
    audio_data = []
    # 数据文件命名格式示例："M003_002_0_0"，其中 M003 表示身份
    identities_list = [
        "W019", "W021", "W023", "W024", "W025", "W026", "W028", "W029",
        "M003", "M005", "M007", "M009", "M011", "M012", "M013", "M019",
        "M022", "M023", "M024", "M025", "M026", "M027", "M028", "M029",
        "M030", "M031", "W009", "W011", "W014", "W015", "W016", "W018",
        # "W019", "W021", "W023", "W024", "W025", "W026", "W028", "W029",
    ]

    if load_audio:  # 第二阶段，加载音频数据
        for key in motion_key:
            if len(keys) >= max_data:
                keys = keys[:max_data]
                motion_data = motion_data[:max_data]
                shape_data = shape_data[:max_data]
                audio_data = audio_data[:max_data]
                break

            if key in audio_key:
                key_split = key.split("_")
                identity = key_split[0]
                load_key = None
                num_ids = len(identities_list)
                
                # 根据身份在 identities_list 中的位置划分数据
                if split == "train":
                    if identity in identities_list[:int(num_ids * 0.7)]:
                        load_key = key
                        keys.append(key)
                elif split == "val":
                    if identity in identities_list[int(num_ids * 0.7):int(num_ids * 0.85)]:
                        load_key = key
                        keys.append(key)
                elif split == "test":
                    if identity in identities_list[int(num_ids * 0.85):]:
                        load_key = key
                        keys.append(key)

                if load_key is not None:
                    m_index = motion_key.index(load_key)
                    m_dir = motion_dir[m_index]
                    m_npy = np.load(m_dir)

                    # 保存运动数据
                    exp = np.squeeze(m_npy[:, :, 300:350])
                    jaw = np.squeeze(m_npy[:, :, 400:])
                    normalized_exp = exp_scaler.transform(exp)
                    normalized_jaw = jaw_scaler.transform(jaw)
                    normalized_exp_jaw = np.concatenate((normalized_exp, normalized_jaw), axis=1)
                    motion_data.append(torch.from_numpy(normalized_exp_jaw).unsqueeze(0))
                    # 保存形状数据
                    shape_data.append(torch.from_numpy(m_npy[:, :, :300]))
                    # 保存音频数据
                    a_index = audio_key.index(key)
                    a_dir = audio_dir[a_index]

                    speech_array, _ = librosa.load(a_dir, sr=16000)
                    speech_array = audio_normalize(speech_array)
                    audio_data.append(speech_array)
                else:
                    pass
            else:
                print(f"No matching audio file for {key}")
                pass

        return keys, motion_data, shape_data, audio_data
    else:     # first stage
        for dir in motion_dir:
            if len(keys) >= max_data:
                keys = keys[:max_data]
                motion_data = motion_data[:max_data]
                shape_data = shape_data[:max_data]
                break

            m_npy = np.load(dir)
            # save file name as key
            keys.append(dir.stem)
            # save motion data
            exp = np.squeeze(m_npy[:, :, 300:350])
            jaw = np.squeeze(m_npy[:, :, 400:])
            normalized_exp = exp_scaler.transform(exp)
            normalized_jaw = jaw_scaler.transform(jaw)
            normalized_exp_jaw = np.concatenate((normalized_exp, normalized_jaw), axis=1)
            motion_data.append(torch.from_numpy(normalized_exp_jaw).unsqueeze(0))
            # save shape data
            shape_data.append(torch.from_numpy(m_npy[:, :, :300]))
        return keys, motion_data, shape_data


