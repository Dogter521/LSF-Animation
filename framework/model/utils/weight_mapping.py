"""
权重映射工具，用于将旧模型权重映射到新架构
"""
import torch
import torch.nn as nn
from typing import Dict, Any

def map_old_weights_to_new(old_state_dict: Dict[str, Any], new_model: nn.Module) -> Dict[str, Any]:
    """
    将旧模型权重映射到新架构
    
    Args:
        old_state_dict: 旧模型的state_dict
        new_model: 新模型实例
    
    Returns:
        映射后的state_dict
    """
    new_state_dict = {}
    
    # 获取新模型的state_dict作为模板
    new_model_dict = new_model.state_dict()
    
    # 权重映射规则
    mapping_rules = {
        # 特征映射层
        'feature_predictor.audio_feature_map.weight': 'feature_predictor.audio_feat.weight',
        'feature_predictor.audio_feature_map.bias': 'feature_predictor.audio_feat.bias',
        'feature_predictor.emotion_feature_map.weight': 'feature_predictor.emo_feat.weight',
        'feature_predictor.emotion_feature_map.bias': 'feature_predictor.emo_feat.bias',
        'feature_predictor.style_embedding.weight': 'feature_predictor.style_proj.weight',
        
        # 压缩层
        'feature_predictor.squasher.0.0.weight': 'feature_predictor.squash_a.0.weight',
        'feature_predictor.squasher.0.0.bias': 'feature_predictor.squash_a.0.bias',
        
        # 位置编码和线性嵌入
        'feature_predictor.encoder_pos_embedding.pe': 'feature_predictor.pos_enc.pe',
        'feature_predictor.encoder_linear_embedding.net.weight': 'feature_predictor.lin_embed.net.weight',
        'feature_predictor.encoder_linear_embedding.net.bias': 'feature_predictor.lin_embed.net.bias',
    }
    
    # 应用映射规则
    for old_key, new_key in mapping_rules.items():
        if old_key in old_state_dict and new_key in new_model_dict:
            # 检查形状是否兼容
            old_shape = old_state_dict[old_key].shape
            new_shape = new_model_dict[new_key].shape
            
            if old_shape == new_shape:
                new_state_dict[new_key] = old_state_dict[old_key]
            else:
                print(f"形状不匹配: {old_key} {old_shape} -> {new_key} {new_shape}")
    
    # 处理transformer层权重
    for i in range(12):  # 假设有12层
        # 音频transformer层
        audio_old_prefix = f'feature_predictor.encoder_transformer_layers.{i}'
        audio_new_prefix = f'feature_predictor.audio_blocks.{i}'
        
        # 情感transformer层
        emo_old_prefix = f'feature_predictor.encoder_transformer_emotion_layers.{i}'
        emo_new_prefix = f'feature_predictor.emo_blocks.{i}'
        
        # 映射transformer权重
        map_transformer_weights(old_state_dict, new_state_dict, audio_old_prefix, audio_new_prefix)
        map_transformer_weights(old_state_dict, new_state_dict, emo_old_prefix, emo_new_prefix)
    
    # 处理融合层权重
    for i in range(12):
        fusion_old_prefix = f'feature_predictor.MHCA.{i}'
        fusion_new_prefix = f'feature_predictor.fusion_blocks.{i}'
        map_fusion_weights(old_state_dict, new_state_dict, fusion_old_prefix, fusion_new_prefix)
    
    # 初始化缺失的权重
    for key, tensor in new_model_dict.items():
        if key not in new_state_dict:
            new_state_dict[key] = tensor.clone()
            print(f"初始化缺失权重: {key}")
    
    return new_state_dict

def map_transformer_weights(old_dict, new_dict, old_prefix, new_prefix):
    """映射transformer层权重"""
    # 注意力层权重
    attn_mappings = {
        f'{old_prefix}.net.0.fn.norm.weight': f'{new_prefix}.net.0.fn.norm.weight',
        f'{old_prefix}.net.0.fn.norm.bias': f'{new_prefix}.net.0.fn.norm.bias',
        f'{old_prefix}.net.0.fn.fn.to_qkv.weight': f'{new_prefix}.net.0.fn.fn.to_qkv.weight',
        f'{old_prefix}.net.0.fn.fn.to_out.weight': f'{new_prefix}.net.0.fn.fn.to_out.weight',
        f'{old_prefix}.net.0.fn.fn.to_out.bias': f'{new_prefix}.net.0.fn.fn.to_out.bias',
    }
    
    # MLP层权重
    mlp_mappings = {
        f'{old_prefix}.net.1.fn.norm.weight': f'{new_prefix}.net.1.fn.norm.weight',
        f'{old_prefix}.net.1.fn.norm.bias': f'{new_prefix}.net.1.fn.norm.bias',
        f'{old_prefix}.net.1.fn.fn.l1.weight': f'{new_prefix}.net.1.fn.fn.l1.weight',
        f'{old_prefix}.net.1.fn.fn.l1.bias': f'{new_prefix}.net.1.fn.fn.l1.bias',
        f'{old_prefix}.net.1.fn.fn.l2.weight': f'{new_prefix}.net.1.fn.fn.l2.weight',
        f'{old_prefix}.net.1.fn.fn.l2.bias': f'{new_prefix}.net.1.fn.fn.l2.bias',
    }
    
    for old_key, new_key in {**attn_mappings, **mlp_mappings}.items():
        if old_key in old_dict and new_key in new_dict:
            if old_dict[old_key].shape == new_dict[new_key].shape:
                new_dict[new_key] = old_dict[old_key]

def map_fusion_weights(old_dict, new_dict, old_prefix, new_prefix):
    """映射融合层权重"""
    # 这里需要根据具体的融合层架构来实现
    # 由于架构差异较大，可能需要手动调整
    pass

def load_model_with_weight_mapping(model_path: str, new_model: nn.Module) -> nn.Module:
    """
    加载模型并应用权重映射
    
    Args:
        model_path: 旧模型权重路径
        new_model: 新模型实例
    
    Returns:
        加载权重后的新模型
    """
    # 加载旧权重
    old_state_dict = torch.load(model_path, map_location='cpu')
    
    # 映射权重
    new_state_dict = map_old_weights_to_new(old_state_dict, new_model)
    
    # 加载到新模型
    new_model.load_state_dict(new_state_dict, strict=False)
    
    return new_model
