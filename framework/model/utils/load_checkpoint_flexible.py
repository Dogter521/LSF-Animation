"""
灵活的模型加载工具，处理权重不匹配的情况
"""
import torch
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def load_checkpoint_flexible(model, ckpt_path: str, eval_mode: bool = True, 
                           device: str = 'cpu', strict: bool = False) -> None:
    """
    灵活加载模型权重，处理权重不匹配的情况
    
    Args:
        model: 要加载权重的模型
        ckpt_path: 权重文件路径
        eval_mode: 是否设置为评估模式
        device: 设备
        strict: 是否严格匹配权重
    """
    try:
        # 尝试正常加载
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # 过滤掉不匹配的权重
        model_state_dict = model.state_dict()
        filtered_state_dict = {}
        
        matched_keys = []
        missing_keys = []
        unexpected_keys = []
        
        for key, value in state_dict.items():
            if key in model_state_dict:
                if model_state_dict[key].shape == value.shape:
                    filtered_state_dict[key] = value
                    matched_keys.append(key)
                else:
                    logger.warning(f"形状不匹配的权重: {key} - 模型: {model_state_dict[key].shape}, 权重: {value.shape}")
                    missing_keys.append(key)
            else:
                unexpected_keys.append(key)
        
        # 检查缺失的权重
        for key in model_state_dict.keys():
            if key not in filtered_state_dict:
                missing_keys.append(key)
        
        logger.info(f"成功匹配的权重: {len(matched_keys)}")
        logger.info(f"缺失的权重: {len(missing_keys)}")
        logger.info(f"意外的权重: {len(unexpected_keys)}")
        
        if missing_keys:
            logger.warning(f"缺失的权重键: {missing_keys[:10]}...")  # 只显示前10个
        if unexpected_keys:
            logger.warning(f"意外的权重键: {unexpected_keys[:10]}...")  # 只显示前10个
        
        # 加载过滤后的权重
        model.load_state_dict(filtered_state_dict, strict=False)
        
        if eval_mode:
            model.eval()
            
        logger.info(f"成功加载模型权重: {ckpt_path}")
        
    except Exception as e:
        logger.error(f"加载模型权重失败: {e}")
        raise

def load_checkpoint_with_fallback(model, ckpt_path: str, eval_mode: bool = True, 
                                device: str = 'cpu') -> bool:
    """
    尝试加载模型权重，如果失败则使用随机初始化
    
    Args:
        model: 要加载权重的模型
        ckpt_path: 权重文件路径
        eval_mode: 是否设置为评估模式
        device: 设备
    
    Returns:
        是否成功加载权重
    """
    try:
        load_checkpoint_flexible(model, ckpt_path, eval_mode, device)
        return True
    except Exception as e:
        logger.warning(f"无法加载预训练权重: {e}")
        logger.info("使用随机初始化的权重")
        
        if eval_mode:
            model.eval()
        return False
