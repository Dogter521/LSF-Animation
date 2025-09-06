import os
import torch
from torch import Tensor
import torch.nn as nn
from typing import List
from hydra.utils import instantiate
import logging
from torch.nn.functional import mse_loss

from framework.model.utils.tools import load_checkpoint, create_one_hot, resample_input, get_shape_para
from framework.model.utils.load_checkpoint_flexible import load_checkpoint_flexible
from framework.model.metrics.compute import ComputeMetrics
from framework.model.base import BaseModel
from framework.data.utils import get_split_keyids
from framework.model.feature_extractor.emotion2vec.scripts import extract_features
import torch.nn.functional as F
import fairseq
import numpy as np

logger = logging.getLogger(__name__)


class VqvaePredict(BaseModel):
    def __init__(self,
                 # passed trough datamodule
                 nfeats: int,
                 split_path: str,
                 one_hot_dim: List,
                 resumed_training: bool,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.nfeats = nfeats
        self.resumed_training = resumed_training
        self.working_dir = self.hparams.working_dir

        self.feature_extractor = instantiate(self.hparams.feature_extractor)
        # print('##################################################################\n\tmamba\n###########################################################')
        # logger.info(f"1. Audio feature extractor '{self.feature_extractor.hparams.name}' loaded")
        self.audio_encoded_dim = self.feature_extractor.audio_encoded_dim    # 768
        # self.iden_dict = np.load('/data/lx22/manba/face/work_25/ProbTalk3D_iden/identity_processed.npy', allow_pickle=True).item()
        # # 添加 emotion_extractor 层
        # self.emotion_extractor = nn.Sequential(
        #     nn.Linear(self.audio_encoded_dim, 128),  # 降低特征维度
        #     nn.ReLU(),
        #     nn.Linear(128, 11),                # 输出维度为 11
        #     nn.Softmax(dim=-1)                 # 转化为概率分布
        # )
        # logger.info(f"Emotion extractor updated to process [B, T, {self.audio_encoded_dim}] and output [B, T, 11]")

        
        
        # Style one-hot embedding
        self.all_identity_list = get_split_keyids(path=split_path, split="train")
        
        self.all_identity_onehot = torch.eye(len(self.all_identity_list))
        # logging.info(f"all_identity_onehot: {self.all_identity_onehot}")


        
        self.identity_extractor = nn.Sequential(
            nn.Linear(300, 256),  # 降低特征维度
            nn.ReLU(),
            nn.Linear(256, 128),  # 输出最终128维特征
            nn.ReLU(),             # 可保留，也可以去掉视情况而定
            nn.Linear(128, len(self.all_identity_list)),                # 输出维度为 11
            nn.Softmax(dim=-1)                 # 转化为概率分布
        )


        

        model_dir = '/data/lx22/manba/face/work_25/ProbTalk3D_iden/framework/model/feature_extractor/emotion2vec/upstream'
        model_path = extract_features.UserDirModule(model_dir)
        fairseq.utils.import_user_module(model_path)
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(['/data/lx22/manba/face/work_25/ProbTalk3D_iden/framework/model/feature_extractor/emotion2vec/pretrained/emotion2vec_base.pt'])
        self.audio_model = model[0]
        for param in self.audio_model.parameters():
            param.requires_grad = False  # 关闭梯度计算
        self.audio_model.eval()   

        # Load motion prior
        self.motion_prior = instantiate(self.hparams.motion_prior,
                                        nfeats=self.hparams.nfeats,
                                        logger_name="none",
                                        resumed_training=False,
                                        _recursive_=False)
        logger.info(f"2. '{self.motion_prior.hparams.modelname}' loaded")
        # Load the motion prior in eval mode
        if os.path.exists(self.hparams.ckpt_path_prior):
            try:
                load_checkpoint(model=self.motion_prior,
                                ckpt_path=self.hparams.ckpt_path_prior,
                                eval_mode=True,
                                device=self.device)
            except Exception as e:
                logger.warning(f"无法加载motion prior权重: {e}")
                logger.info("使用随机初始化的motion prior权重")
        else:
            raise ValueError(f"Motion Autoencoder path not found: {self.hparams.ckpt_path_prior}")
        for param in self.motion_prior.parameters():
            param.requires_grad = False

        self.feature_predictor = instantiate(self.hparams.feature_predictor,
                                             audio_dim=self.audio_encoded_dim * 2,   # 768*2
                                             emotion_dim = self.audio_encoded_dim * 2,
                                             one_hot_dim=len(self.all_identity_list))   # 32+8+3
        logger.info(f"3. 'Audio Encoder' loaded")

        self.optimizer = instantiate(self.hparams.optim, params=self.parameters())
        self._losses = torch.nn.ModuleDict({split: instantiate(self.hparams.losses, _recursive_=False)
                                            for split in ["losses_train", "losses_test", "losses_val"]})

        self.losses = {key: self._losses["losses_" + key] for key in ["train", "test", "val"]}
        self.metrics = ComputeMetrics()

        # If we want to override it at testing time
        self.temperature = 0.2
        self.k = 1

        self.__post_init__()

    # Forward: audio => motion, called during sampling
    # extract emotion and intensity from audio feature by using emotion_extractor

    def forward(self, batch, sample, generation=True) -> Tensor:
        self.feature_predictor.to(self.device)
        self.feature_extractor.to(self.device)
        self.motion_prior.to(self.device)
        self.audio_model.to(self.device)
        
        shape_params = batch['shape'][0]

        shape_params = shape_params.to(self.device)

        identity_input = shape_params[:, 0, :300]  # [B, 300]

        # 通过 identity_extractor 网络
        identity_probs = self.identity_extractor(identity_input)  # [B, len(self.all_identity_list)]

        
        # audio feature extraction
        audio_feature = self.feature_extractor(batch['audio'], False)  # list of [B, Ts, 768]

        
        # 提取 emotion 和 intensity one-hot 编码
        audio_data = torch.from_numpy(batch['audio'][0]).float().to(self.device)
        audio_data = F.layer_norm(audio_data, audio_data.shape)
        audio_data = audio_data.view(1, -1)
        emotion_feature = self.audio_model.extract_features(audio_data)['x'] # [B, T, 768]
        # emotion_logits = self.emotion_extractor(emotion_feature['x'])  # [B, T, 11]
        style_one_hot = identity_probs
        


        resample_audio_feature = []
        resample_emotion_feature = []

        for idx in range(len(audio_feature)):
            # === audio ===
            audio_feature_one = audio_feature[idx]
            if audio_feature_one.shape[1] % 2 != 0:
                audio_feature_one = audio_feature_one[:, :audio_feature_one.shape[1] - 1, :]

            if not generation:
                if audio_feature_one.shape[1] > batch['motion'][0].shape[1] * 2:
                    print("[audio] Shape checking:", audio_feature_one.shape, batch['motion'][0].shape)
                    audio_feature_one = audio_feature_one[:, :batch['motion'][0].shape[1] * 2, :]

            audio_feature_one = torch.reshape(audio_feature_one,
                                            (1, audio_feature_one.shape[1] // 2, audio_feature_one.shape[2] * 2))
            resample_audio_feature.append(audio_feature_one)

            # === emotion ===
            emotion_feature_one = emotion_feature
            if emotion_feature_one.shape[1] % 2 != 0:
                emotion_feature_one = emotion_feature_one[:, :emotion_feature_one.shape[1] - 1, :]

            if not generation:
                if emotion_feature_one.shape[1] > batch['motion'][0].shape[1] * 2:
                    print("[emotion] Shape checking:", emotion_feature_one.shape, batch['motion'][0].shape)
                    emotion_feature_one = emotion_feature_one[:, :batch['motion'][0].shape[1] * 2, :]

            emotion_feature_one = torch.reshape(emotion_feature_one,
                                                (1, emotion_feature_one.shape[1] // 2, emotion_feature_one.shape[2] * 2))
            resample_emotion_feature.append(emotion_feature_one)

        assert len(resample_audio_feature) == 1, "Batch size > 1 not supported"
        assert len(resample_emotion_feature) == 1, "Batch size > 1 not supported"


        # only works for batch_size=1
        batch['audio'] = torch.cat(resample_audio_feature, dim=0)
        emotion_features = torch.cat(resample_emotion_feature, dim=0)
        prediction = self.feature_predictor(batch['audio'], emotion_features, style_one_hot)  # [B, T, 256]
        motion_quant_pred, _, _ = self.motion_prior.quantize(prediction, sample=sample,
                                                             temperature=self.temperature,  # 0.2 by default,
                                                             k=self.k)      # 1 by default,  
        motion_out = self.motion_prior.motion_decoder(motion_quant_pred)    # [B, T, 53]

        return motion_out

    # Called during training
    def allsplit_step(self, split: str, batch, batch_idx):
        # extract audio features
        audio_feature = self.feature_extractor(batch['audio'], False)       # list of [1, Ts, 768]
        shape_params = batch['shape'][0]
        shape_params = shape_params.to(self.device)
        identity_input = shape_params[:, 0, :]  # [B, 300]
        gt_motion = batch['motion'][0] # [B, T, 53]
        # logger.info(gt_motion)
        # logger.info(gt_motion.shape)

        # 通过 identity_extractor 网络
        identity_probs = self.identity_extractor(identity_input)  # [B, len(self.all_identity_list)]
        # 提取 emotion 和 intensity one-hot 编码
        audio_data = torch.from_numpy(batch['audio'][0]).float().to(self.device)
        audio_data = F.layer_norm(audio_data, audio_data.shape)
        audio_data = audio_data.view(1, -1)
        emotion_feature = self.audio_model.extract_features(audio_data)['x'] # [B, T, 768]
        # emotion_one_hot = emotion_feature['x'].mean(dim=1)       # 对时间维度求平均
        
        # style_one_hot = torch.cat([identity_probs, emotion_one_hot], dim=-1)
        style_one_hot = identity_probs


        resample_audio_feature = []
        resample_motion_feature = []
        resample_emotion_feature = []
        
        # logger.info(f"Batch shape: {audio_feature[0].shape}, {batch['motion'][0].shape},{emotion_feature.shape}")
        for idx in range(len(audio_feature)):
            resample_audio, resample_motion, resample_emotion = resample_input(audio_feature[idx], batch['motion'][idx],
                                                             self.feature_extractor.hparams.output_framerate,
                                                             self.hparams.video_framerate,emotion_feature)
            resample_audio_feature.append(resample_audio)
            resample_motion_feature.append(resample_motion)
            resample_emotion_feature.append(resample_emotion)
        assert len(resample_audio_feature) == 1, "Batch size > 1 not supported"

        

        # only works for batch_size=1
        batch['audio'] = torch.cat(resample_audio_feature, dim=0)       # [1, T, 768*2]
        batch['motion'] = torch.cat(resample_motion_feature, dim=0)     # [1, T, 53]
        emotion_features = torch.cat(resample_emotion_feature, dim=0)           # [1, T, 768]
        # logger.info(f"Batch shape: {batch['audio'].shape}, {batch['motion'].shape},{emotion_features.shape}")

        prediction = self.feature_predictor(batch['audio'], emotion_features, style_one_hot)  # [B, T, 256]
        motion_quant_pred, _, _ = self.motion_prior.quantize(prediction)        # [B, T, 256]
        motion_pred = self.motion_prior.motion_decoder(motion_quant_pred)       # [B, T, 53]
        
        motion_quant_ref, _ = self.motion_prior.get_quant(batch['motion'])      # [B, T, 256]
        motion_ref = batch['motion']
        assert motion_pred.shape == motion_ref.shape, "Dimension mismatch between prediction and reference motion."

        # style_alignment_loss = mse_loss(style_one_hot, style_one_hot_gt)
        # logger.info(f"motion_quant_pred.shape: {motion_quant_pred.shape},motion_quant_ref.shape:{motion_quant_ref.shape},motion_pred.shape: {motion_pred.shape}, motion_ref.shape: {motion_ref.shape}")
        motion_loss = self.losses[split].update(motion_quant_pred=motion_quant_pred, motion_quant_ref=motion_quant_ref,
                                         motion_pred=motion_pred, motion_ref=motion_ref) #, gt_motion = gt_motion
        total_loss = motion_loss # + 0.1 * style_alignment_loss
        # self.log(f"style_alignment_loss/{split}", style_alignment_loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"total_loss/{split}", total_loss.item(), on_step=True, on_epoch=True, prog_bar=True)



        # Compute the metrics
        if split == "val":
            self.metrics.update(motion_pred.detach(),
                                motion_ref.detach(),
                                [motion_ref.shape[1]] * motion_ref.shape[0])

        
        # Log the losses
        self.allsplit_batch_end(split, batch_idx)

        # Show loss on progress bar
        if "total/train" in self.trainer.callback_metrics:
            loss_train = self.trainer.callback_metrics["total/train"].item()
            self.log("loss_train", loss_train, prog_bar=True, on_step=True, on_epoch=False)

        if "total/val" in self.trainer.callback_metrics:
            loss_val = self.trainer.callback_metrics["total/val"].item()
            self.log("loss_val", loss_val, prog_bar=True, on_step=True, on_epoch=False)

        return total_loss

