import os
import torch
from collections import defaultdict
from torch.utils import data
import numpy as np
import pickle
import random
from tqdm import tqdm
from transformers import Wav2Vec2Processor
import librosa
import sys


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data, subjects_dict, data_type="train"):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.identities = subjects_dict["all"]          # ← 全部 32
        self.id2idx = {sid: i for i, sid in enumerate(self.identities)}
        self.one_hot_labels = np.eye(len(self.identities))  # 32×32

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        file_name = self.data[index]["name"]
        audio = self.data[index]["audio"]
        vertice = self.data[index]["vertice"]
        template = self.data[index]["template"]
        if self.data_type == "train" or self.data_type == "val" or self.data_type == "test":
            subject = file_name.split("_")[0]
            id_one_hot = torch.from_numpy(self.one_hot_labels[self.id2idx[subject]])
            id_one_hot = torch.tensor(id_one_hot)
            emotion_idx = int(file_name.split("_")[2])
            emotion_one_hot = torch.eye(8)[emotion_idx]
            intensity_idx = int(file_name.split("_")[3][:1])
            intensity_one_hot = torch.eye(3)[intensity_idx]
            one_hot = torch.cat([id_one_hot, emotion_one_hot, intensity_one_hot], dim=0).float()
        else:
            one_hot = self.one_hot_labels

        return torch.FloatTensor(audio), vertice, torch.FloatTensor(template), torch.FloatTensor(
            one_hot), file_name

    def __len__(self):
        return self.len



from pathlib import Path
def read_data(args):
    print("Loading data...")
    sys.stdout.flush()

    # ---------- 原始读取流程（保持不变） ----------
    data = defaultdict(dict)
    audio_path    = os.path.join(args.data_path, args.dataset, args.wav_path)
    vertices_path = os.path.join(args.data_path, args.dataset, args.vertices_path)

    modelpath_processor = (
        "/data/lx22/manba/face/work_25/audio_models/wav2vec_model1/"
        "models--facebook--hubert-xlarge-ls960-ft/snapshots/86a09e67e0c8d074533992379242405825516eca"
    )
    processor = Wav2Vec2Processor.from_pretrained(modelpath_processor)

    template_file = os.path.join(args.data_path, args.template_file)
    with open(template_file, "rb") as fin:
        templates = pickle.load(fin, encoding="latin1")

    for root, dirs, files in os.walk(vertices_path):
        for f in tqdm(files):
            if not f.endswith(".npy"):
                continue

            # 构建样本字典
            m_path       = os.path.join(root, f)
            key          = f.replace(".npy", ".wav")      # 用来索引 data
            subject_id   = f.split("_")[0]
            data[key]["vertice"]  = m_path
            data[key]["name"]     = f
            data[key]["template"] = templates.get(subject_id,
                                    np.zeros(args.vertice_dim)
                                ).reshape(-1)

            wav_path = os.path.join(audio_path, key)
            if not os.path.exists(wav_path):
                # 如果没有对应的 wav，就丢掉这个样本
                del data[key]
                continue

            # 加载并处理音频
            speech_array, sr = librosa.load(wav_path, sr=16000)
            input_values = np.squeeze(
                processor(
                    speech_array,
                    sampling_rate=sr,
                    return_tensors="pt",
                    padding="longest"
                ).input_values
            )
            data[key]["audio"] = input_values

    # ---------- 数据划分逻辑（对齐代码1） ----------
    identities_list = [
        "W019","W021","W023","W024","W025","W026","W028","W029",
        "M003","M005","M007","M009","M011","M012","M013","M019",
        "M022","M023","M024","M025","M026","M027","M028","M029",
        "M030","M031","W009","W011","W014","W015","W016","W018",
    ]
    n_id    = len(identities_list)
    n_train = int(n_id * 0.70)
    n_val   = int(n_id * 0.15)

    id_train = set(identities_list[:n_train])
    id_val   = set(identities_list[n_train:n_train + n_val])
    id_test  = set(identities_list[n_train + n_val:])

    subjects_dict = {
        "train": list(id_train),
        "val":   list(id_val),
        "test":  list(id_test),
        "all":   identities_list
    }

    # 识别数据集名，以便对 vocaset/biwi 做额外的 sentence-based 划分
    dataset_name = os.path.basename(args.dataset).lower()

    train_data, valid_data, test_data = [], [], []
    for key, v in data.items():
        subject_id = key.split("_")[0]

        
        if   subject_id in id_train: train_data.append(v)
        elif subject_id in id_val:   valid_data.append(v)
        elif subject_id in id_test:  test_data.append(v)
        # 否则跳过

    print(
        "number of training data:",   len(train_data),
        "validation data:",           len(valid_data),
        "test data:",                 len(test_data)
    )
    sys.stdout.flush()
    return train_data, valid_data, test_data, subjects_dict



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloaders(args):
    g = torch.Generator()
    g.manual_seed(0)
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args)
    train_data = Dataset(train_data, subjects_dict, "train")
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=1, shuffle=True, worker_init_fn=seed_worker,
                                       generator=g)
    valid_data = Dataset(valid_data, subjects_dict, "val")
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False)
    test_data = Dataset(test_data, subjects_dict, "test")
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    return dataset


if __name__ == "__main__":
    get_dataloaders()

