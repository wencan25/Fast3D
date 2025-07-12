## built-in
import argparse
import logging
import math
import os
import random
import types
import pickle, json
import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"

## third-party
import torch
from torch import optim as optim
from torch.optim import Optimizer
import math
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist
from functools import partial
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, ConcatDataset
from tqdm.auto import tqdm
import copy
import transformers
from tqdm import tqdm
import numpy as np
from easydict import EasyDict
import re
import sys

## own
from fast3d_dataset import Fast3dDataset, fast3d_collator
from modeling_fast3d import Fast3dNet, Fast3dNetConfig
from cmt_module import CMTConfig


logger = get_logger(__name__)
logging.getLogger().handlers = []


def get_yaml_file(file_path):
    import yaml

    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def parse_args():
    parser = argparse.ArgumentParser()
    ## adding args here for more control from CLI is possible
    parser.add_argument("--config_file", default="config/debug_train.yaml")
    args = parser.parse_args()
    yaml_config = get_yaml_file(args.config_file)
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    yaml_config.update(args_dict)
    args = EasyDict(yaml_config)
    return args, yaml_config


##########################################
def setup_dataloaders(args: EasyDict):
    train_val_loaders = []
    for tags, split in zip([args.train_tags, args.val_tags], ["train", "val"]):
        set_names = tags.split("#")
        datasets = {}
        for set_name in set_names:
            anno_file = os.path.join(
                args.annotation_root, f"{set_name}_mask3d_{split}.json"
            )
            if not os.path.isfile(anno_file):
                anno_file = os.path.join(args.annotation_root, f"{set_name}_{split}.json")
            attn_maps_file = os.path.join(
                args.attn_maps_root, f"infer_attn_maps_{split}_{set_name}.pt"
            )
            attibutes_file = os.path.join(
                args.annotation_root, f"scannet_mask3d_{split}_attributes.pt"
            )
            dataset = Fast3dDataset(
                anno_file,
                attn_maps_file,
                attibutes_file,
                use_ori_attn_maps=args.use_ori_attn_maps,
                use_mentioned_oids_in_answers=args.use_mentioned_oids_in_answers,
            )
            datasets[set_name] = dataset
        if split == "train":
            datasets = {tags: ConcatDataset(datasets.values())}
        dataloaders = {}
        for k, v in datasets.items():
            dataloaders[k] = DataLoader(
                v,
                batch_size=args.train_batch_size
                if split == "train"
                else args.eval_batch_size,
                shuffle=True if split == "train" else False,
                collate_fn=fast3d_collator,
                num_workers=args.num_workers,
            )
        train_val_loaders.append(dataloaders)
    return train_val_loaders


def get_num_params(model):
    import numpy as np

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))
    return num_params


def get_optimizer(model: Fast3dNet, args: EasyDict):
    optimizer_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if "text_encoder" in name:
            lr = args.text_lr
        else:
            lr = args.lr
        if len(param.shape) == 1 or name.endswith(".bias"):
            weight_decay = 0
        else:
            weight_decay = args.weight_decay
        optimizer_params.append({"params": param, "lr": lr, "weight_decay": weight_decay})
    optimizer = optim.AdamW(optimizer_params, betas=(0.9, 0.999), fused=True)
    return optimizer


# sched utils
def get_scheduler(optimizer, num_warmup_steps, num_training_steps, args: EasyDict):
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=0.5,
        min_lr_multi=args.min_lr_multi,
    )
    return lr_scheduler


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr_multi: float = 0.0,
    last_epoch: int = -1,
):
    """
    Modified from https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/optimization.py

    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        min_lr_multi (`float`, *optional*, defaults to 0):
            The minimum learning rate multiplier. Thus the minimum learning rate is base_lr * min_lr_multi.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return max(
                min_lr_multi, float(current_step) / float(max(1, num_warmup_steps))
            )
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            min_lr_multi,
            0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


@torch.no_grad()
def validate_during_training(model, val_dataloader, accelerator, save_path=None):
    model.eval()
    total_loss = {}
    attn_maps = {}
    for batch in tqdm(
        val_dataloader,
        total=len(val_dataloader),
        disable=not accelerator.is_local_main_process,
    ):
        outputs = model(**batch)
        for k, v in outputs.items():
            if "loss" in k:
                total_loss.setdefault(k, [])
                total_loss[k].append(v.detach().float().cpu())

        if save_path is not None:
            for i, attn_map in zip(batch["index"], outputs["logits"]):
                attn_maps[i] = {
                    "a_map": attn_map.detach().float().cpu(),
                }

        from sklearn.metrics import f1_score

        for k in [5, 10, 20]:
            # Convert logits to probabilities
            logits = outputs["logits"]
            predictions = torch.zeros_like(logits)
            topk_values, topk_indices = torch.topk(logits, k=k, dim=-1)
            predictions.scatter_(1, topk_indices, 1)
            predictions = predictions.cpu().int().numpy()
            # Calculate F1 score
            topk_values, topk_indices = torch.topk(batch["attn_maps"], k=k, dim=-1)
            targets = torch.zeros_like(batch["attn_maps"])
            targets.scatter_(1, topk_indices, 1)
            targets = targets.cpu().int().numpy()
            total_loss.setdefault(f"topk_{k}_f1_score", [])
            for i in range(targets.shape[0]):
                f1 = f1_score(targets[i], predictions[i], average=None)[1]
                total_loss[f"topk_{k}_f1_score"].append(f1)
    model.train()
    if accelerator.use_distributed and accelerator.num_processes > 1:
        all_ranks_objects = [None for _ in range(accelerator.num_processes)]
        dist.all_gather_object(all_ranks_objects, total_loss)
        total_loss = {}
        for rank_loss in all_ranks_objects:
            for k, v in rank_loss.items():
                total_loss.setdefault(k, [])
                total_loss[k] += v

        if save_path is not None:
            all_ranks_objects = [None for _ in range(accelerator.num_processes)]
            dist.all_gather_object(all_ranks_objects, attn_maps)
            attn_maps = {}
            for rank_attn_maps in all_ranks_objects:
                for i, attn_map in rank_attn_maps.items():
                    attn_maps[i] = attn_map

    if save_path is not None and accelerator.is_local_main_process:
        torch.save(attn_maps, save_path)

    for k, v in total_loss.items():
        total_loss[k] = float(np.mean(v))
    return total_loss


def main():
    args, args_dict = parse_args()
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=args.output_dir,
        kwargs_handlers=[kwargs],
    )

    if accelerator.is_local_main_process:
        print(args)
        log_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.eval_only:
            project_name = f"eval_fast3d-{log_time}"
        else:
            project_name = f"train_fast3d-{log_time}"
        accelerator.init_trackers(
            project_name=project_name,
            config=args_dict,
        )
        # Check if tensorboard tracker is actually available
        tracker = accelerator.get_tracker("tensorboard")
        LOG_DIR = os.path.join(args.output_dir, project_name)
        if accelerator.use_distributed and accelerator.num_processes > 1:
            print("rank", dist.get_rank(), f"TensorBoard log directory: {LOG_DIR}")
    else:
        LOG_DIR = ""

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=True)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
    accelerator.wait_for_everyone()

    # debug
    logger.info("***** Debug logging *****")
    if accelerator.use_distributed and accelerator.num_processes > 1:
        print("rank", dist.get_rank(), args)
        print("rank", dist.get_rank(), "hello")
        print("rank", dist.get_rank(), "bye")

    # get model
    model_config = Fast3dNetConfig(
        # TODO: add other hyperparams here
        mm_encoder=CMTConfig(
            spatial_dec=args.use_spatial_dec,
        ),
        train_text_encoder=args.train_text_encoder,
    )
    model = Fast3dNet(model_config)
    # model parameters
    params = get_num_params(model) / 1e6
    model = model.train()

    if args.eval_only:
        model.load_state_dict(torch.load(args.pretrained_model_path)["model_state_dict"])

    # get dataloaders
    train_dataloader, val_dataloaders = setup_dataloaders(args)
    train_dataloader = list(train_dataloader.values())[0]

    # get optimizer
    optimizer = get_optimizer(model, args)

    # prepare for accelerator
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    for k, v in val_dataloaders.items():
        val_dataloaders[k] = accelerator.prepare(v)

    if args.eval_only:
        for val_set_name, val_dataloader in val_dataloaders.items():
            logger.info(f"Evaluating {val_set_name}...")
            save_path = (
                os.path.join(LOG_DIR, f"infer_attn_maps_val_{val_set_name}.pt")
                if args.save_attn_maps
                else None
            )
            val_results = validate_during_training(
                model, val_dataloader, accelerator, save_path=save_path
            )
            if accelerator.is_local_main_process:
                print(json.dumps(val_results, indent=4))
        return

    NUM_UPDATES_PER_EPOCH = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    MAX_TRAIN_STEPS = NUM_UPDATES_PER_EPOCH * args.max_epoch
    MAX_TRAIN_EPOCHS = math.ceil(MAX_TRAIN_STEPS / NUM_UPDATES_PER_EPOCH)
    TOTAL_TRAIN_BATCH_SIZE = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    EVAL_STEPS = (
        args.val_interval
        if isinstance(args.val_interval, int)
        else int(args.val_interval * NUM_UPDATES_PER_EPOCH)
    )

    # get lr_scheduler
    lr_scheduler = get_scheduler(
        optimizer,
        num_warmup_steps=int(args.warmup_epochs * NUM_UPDATES_PER_EPOCH),
        num_training_steps=MAX_TRAIN_STEPS,
        args=args,
    )

    logger.info("***** Running training *****")
    logger.info(f"  Model trainable parameters = {params}M")
    logger.info(f"  Num train examples = {len(train_dataloader) * args.train_batch_size}")
    for name, loader in val_dataloaders.items():
        logger.info(f"  Num dev examples {name} = {len(loader) * args.eval_batch_size}")
    logger.info(f"  Num Epochs = {MAX_TRAIN_EPOCHS}")
    logger.info(f"  Per device train batch size = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {TOTAL_TRAIN_BATCH_SIZE}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {MAX_TRAIN_STEPS}")
    logger.info(f"  Per device eval batch size = {args.eval_batch_size}")
    logger.info(f"  Eval steps = {EVAL_STEPS}")

    # training loop
    completed_steps = 0
    logging_interval_loss = {}
    total_loss = {}

    progress_bar = tqdm(
        range(MAX_TRAIN_STEPS), disable=not accelerator.is_local_main_process, ncols=100
    )

    save_metric_key = "val_scanrefer_topk_5_f1_score"
    save_metric_value = None
    save_metric_larger_better = True

    for epoch in range(MAX_TRAIN_EPOCHS):
        set_seed(args.seed + epoch)
        progress_bar.set_description(f"epoch: {epoch + 1}")
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    outputs = model(**batch)
                for k, v in outputs.items():
                    if "loss" in k:
                        logging_interval_loss.setdefault(k, 0)
                        logging_interval_loss[k] += v.detach().float()
                loss = outputs["loss"]
                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                ## one optimization step
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1
                    if not accelerator.optimizer_step_was_skipped:
                        lr_scheduler.step()

                    if args.logging_steps and completed_steps % args.logging_steps == 0:
                        avg_loss = {}
                        for k, v in logging_interval_loss.items():
                            avg_loss[k] = (
                                accelerator.gather(v).mean().item()
                                / args.gradient_accumulation_steps
                                / args.logging_steps
                            )
                        for k, v in logging_interval_loss.items():
                            total_loss.setdefault(k, 0)
                            total_loss[k] += (
                                accelerator.gather(v).mean().item()
                                / args.gradient_accumulation_steps
                            )

                        lr_string = f"{lr_scheduler.get_last_lr()[-1]:.6f}"
                        loss_string = f"{avg_loss['loss']:.6f}"
                        progress_bar.set_postfix(loss=loss_string, lr=lr_string)

                        to_be_logged = {
                            "learning_rate": lr_scheduler.get_last_lr()[-1],
                        }
                        for k, v in avg_loss.items():
                            to_be_logged[f"train_{k}"] = v
                        for k, v in total_loss.items():
                            to_be_logged[f"rolling_{k}"] = v / completed_steps
                        accelerator.log(to_be_logged, step=completed_steps)
                        logging_interval_loss = {}

                    if completed_steps % EVAL_STEPS == 0:
                        # evaluate
                        all_val_results = {}
                        if args.eval_train_set:
                            logger.info("Evaluating train set...")
                            train_results = validate_during_training(
                                model, train_dataloader, accelerator
                            )
                            for k, v in train_results.items():
                                all_val_results[f"val_train_{k}"] = float(v)

                        for val_set_name, val_dataloader in val_dataloaders.items():
                            logger.info(f"Evaluating {val_set_name}...")
                            val_results = validate_during_training(
                                model, val_dataloader, accelerator
                            )
                            for k, v in val_results.items():
                                all_val_results[f"val_{val_set_name}_{k}"] = float(v)
                        accelerator.log(all_val_results, step=completed_steps)

                        # save model
                        if accelerator.is_local_main_process:
                            unwrapped_model = accelerator.unwrap_model(model)
                            save_dict = {
                                "step": completed_steps,
                                "optimizer_state_dict": optimizer.state_dict(),
                                "model_state_dict": unwrapped_model.state_dict(),
                            }
                            torch.save(
                                save_dict, os.path.join(LOG_DIR, "checkpoint_last.pth")
                            )
                            if save_metric_key in all_val_results:
                                if save_metric_larger_better:
                                    if (
                                        save_metric_value is None
                                        or all_val_results[save_metric_key]
                                        > save_metric_value
                                    ):
                                        save_metric_value = all_val_results[
                                            save_metric_key
                                        ]
                                        torch.save(
                                            save_dict,
                                            os.path.join(LOG_DIR, "checkpoint_best.pth"),
                                        )
                                else:
                                    if (
                                        save_metric_value is None
                                        or all_val_results[save_metric_key]
                                        < save_metric_value
                                    ):
                                        save_metric_value = all_val_results[
                                            save_metric_key
                                        ]
                                        torch.save(
                                            save_dict,
                                            os.path.join(LOG_DIR, "checkpoint_best.pth"),
                                        )

    if accelerator.is_local_main_process:
        tracker.finish()
    accelerator.end_training()


if __name__ == "__main__":
    main()
