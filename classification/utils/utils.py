# --------------------------------------------------------
# Modified By $@#Anonymous#@$
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
from math import inf
import torch
import torch.distributed as dist
from timm.utils import ModelEma as ModelEma

def load_checkpoint_ema(config, model, optimizer, lr_scheduler, loss_scaler, logger, model_ema: ModelEma=None):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    
    if 'model' in checkpoint:
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        logger.info(f"resuming model: {msg}")
    else:
        logger.warning(f"No 'model' found in {config.MODEL.RESUME}! ")

    if model_ema is not None:
        if 'model_ema' in checkpoint:
            msg = model_ema.ema.load_state_dict(checkpoint['model_ema'], strict=False)
            logger.info(f"resuming model_ema: {msg}")
        else:
            logger.warning(f"No 'model_ema' found in {config.MODEL.RESUME}! ")

    max_accuracy = 0.0
    max_accuracy_ema = 0.0
    steps = 0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        if config.MODEL.DDP == 'torch':
            config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']
        if 'max_accuracy_ema' in checkpoint:
            max_accuracy_ema = checkpoint['max_accuracy_ema']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy, max_accuracy_ema, steps


def load_pretrained_ema(config, model, logger, model_ema: ModelEma=None):
    logger.info(f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
    
    if 'model' in checkpoint:
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        logger.warning(msg)
        logger.info(f"=> loaded 'model' successfully from '{config.MODEL.PRETRAINED}'")
    else:
        logger.warning(f"No 'model' found in {config.MODEL.PRETRAINED}! ")

    if model_ema is not None:
        if "model_ema" in checkpoint:
            logger.info(f"=> loading 'model_ema' separately...")
        key = "model_ema" if ("model_ema" in checkpoint) else "model"
        if key in checkpoint:
            msg = model_ema.ema.load_state_dict(checkpoint[key], strict=False)
            logger.warning(msg)
            logger.info(f"=> loaded '{key}' successfully from '{config.MODEL.PRETRAINED}' for model_ema")
        else:
            logger.warning(f"No '{key}' found in {config.MODEL.PRETRAINED}! ")

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint_ema(config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                        logger, model_ema: ModelEma=None, max_accuracy_ema=None,steps=0, ckpt_name=None):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'scaler': loss_scaler.state_dict(),
                  'epoch': epoch,
                  'config': config,
                  'steps': steps}
    
    if model_ema is not None:
        save_state.update({'model_ema': model_ema.ema.state_dict(),
            'max_accuray_ema': max_accuracy_ema})
    if ckpt_name is None:
        save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    else:
        save_path = os.path.join(config.OUTPUT, f'{ckpt_name}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    try:
        checkpoints = os.listdir(output_dir)
    except:
        return None
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    #print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        resume_file = None
        checkpoints = [os.path.join(output_dir, ckpt) for ckpt in checkpoints]
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        for ckpt_path in checkpoints:
            try:
                checkpoint = torch.load(ckpt_path)
                resume_file = ckpt_path
                break
            except Exception as e:
                print(f"Failed to load checkpoint {ckpt_path}: {e}")
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor, ddp = 'torch'):
    rt = tensor.clone()
    if ddp == 'torch':
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

