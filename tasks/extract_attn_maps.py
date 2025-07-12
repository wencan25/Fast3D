import datetime
import logging
import time
from os.path import join

import pandas as pd
import torch
import torch.distributed as dist
import wandb
from torch.utils.data import ConcatDataset
from functools import partial

from dataset import MetaLoader, create_dataset_attn_maps, create_loader, create_sampler
from dataset.dataset_train import train_collate_fn
from dataset.dataset_val import val_collate_fn

from models.chat3d_fast_get_attn_maps import Chat3D

from tasks.shared_utils import get_media_types, setup_model
from utils.basic_utils import MetricLogger, SmoothedValue, setup_seed
from utils.config_utils import setup_main
from utils.distributed import get_rank, get_world_size, is_main_process
from utils.logger import log_dict_to_wandb, setup_wandb

import numpy as np
from tqdm import tqdm

import json
import os

logger = logging.getLogger(__name__)


def evaluate_all(
    model,
    model_without_ddp,
    val_loaders,
    epoch,
    global_step,
    device,
    config,
    is_val_loader=True,
):
    logger.info("Start evaluating...")
    model.eval()
    # model_without_ddp.llama_model.config.use_cache = True
    for val_loader in val_loaders:
        evaluate(model, val_loader, epoch, global_step, device, config, is_val_loader)

    model.train()


def evaluate(model, val_loader, epoch, global_step, device, config, is_val_loader=True):
    eval_name = val_loader.dataset.datasets[0].dataset_name
    logger.info(f"Evaluating {eval_name}...")
    if config.distributed:
        val_loader.sampler.set_epoch(epoch)

    save_preds = []
    logger.info(f"batch-size={val_loader.batch_size} length(#batches)={len(val_loader)}")
    for i, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
        for k in batch.keys():
            if type(batch[k]) == torch.Tensor:
                batch[k] = batch[k].to(device)
        with torch.no_grad():
            pred = model(**batch)
        save_preds += pred

    if len(save_preds) > 0:
        split = "val" if is_val_loader else "train"
        save_path = f"infer_attn_maps_rank{get_rank()}_{split}_{eval_name}.pt"
        torch.save(save_preds, os.path.join(config.output_dir, save_path))

    dist.barrier()
    if is_main_process():
        save_preds = []
        split = "val" if is_val_loader else "train"
        for rank in range(config.gpu_num):
            path = os.path.join(
                config.output_dir, f"infer_attn_maps_rank{rank}_{split}_{eval_name}.pt"
            )
            if os.path.exists(path):
                preds = torch.load(path)
                save_preds += preds
                os.remove(path)
        torch.save(
            save_preds,
            os.path.join(config.output_dir, f"infer_attn_maps_{split}_{eval_name}.pt"),
        )


def setup_dataloaders(config):
    # train datasets, create a list of data loaders
    train_datasets, val_datasets = create_dataset_attn_maps(config)

    if config.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        train_samplers = create_sampler(
            train_datasets, [False] * len(train_datasets), num_tasks, global_rank
        )
        val_samplers = create_sampler(
            val_datasets, [False] * len(val_datasets), num_tasks, global_rank
        )
    else:
        train_samplers = [None] * len(train_datasets)
        val_samplers = [None] * len(val_datasets)

    train_loaders = create_loader(
        train_datasets,
        train_samplers,
        batch_size=[config.batch_size] * len(train_datasets),
        num_workers=[config.num_workers] * len(train_datasets),
        is_trains=[False] * len(train_datasets),
        collate_fns=[train_collate_fn] * len(train_datasets),
    )
    _val_collate_fn = partial(val_collate_fn, use_external_attn_maps=False)
    val_loaders = create_loader(
        val_datasets,
        val_samplers,
        batch_size=[config.batch_size] * len(val_datasets),
        num_workers=[config.num_workers] * len(val_datasets),
        is_trains=[False] * len(val_datasets),
        collate_fns=[_val_collate_fn] * len(val_datasets),
    )

    return train_loaders, val_loaders


def main(config):
    if is_main_process() and config.wandb.enable:
        run = setup_wandb(config)

    # torch.autograd.set_detect_anomaly(True)
    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)

    train_loaders, val_loaders = setup_dataloaders(config)

    num_steps_per_epoch = sum(len(d) for d in train_loaders)
    config.scheduler.num_training_steps = num_steps_per_epoch * config.scheduler.epochs
    config.scheduler.num_warmup_steps = (
        num_steps_per_epoch * config.scheduler.warmup_epochs
    )
    torch.backends.cudnn.benchmark = True

    model_cls = eval(config.model.get("model_cls", "Chat3D"))
    (
        model,
        model_without_ddp,
        optimizer,
        scheduler,
        scaler,
        start_epoch,
        global_step,
    ) = setup_model(
        config,
        model_cls=model_cls,
        find_unused_parameters=True,
    )
    if is_main_process() and config.wandb.enable:
        wandb.watch(model)

    save_step_interval = 1
    start_time = time.time()

    if config.evaluate:
        logger.info("Infer attention maps... train")
        evaluate_all(
            model,
            model_without_ddp,
            train_loaders,
            start_epoch - 1,
            global_step,
            device,
            config,
            is_val_loader=False,
        )
        logger.info("Infer attention maps... val")
        evaluate_all(
            model,
            model_without_ddp,
            val_loaders,
            start_epoch - 1,
            global_step,
            device,
            config,
            is_val_loader=True,
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")
    logger.info(f"Checkpoints and Logs saved at {config.output_dir}")

    if is_main_process() and config.wandb.enable:
        run.finish()


if __name__ == "__main__":
    cfg = setup_main()
    main(cfg)
