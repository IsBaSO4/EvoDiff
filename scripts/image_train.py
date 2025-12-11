import os
import sys
from pathlib import Path
import torch

torch.cuda.empty_cache()

print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

# Get absolute path of current file
current_file = Path(__file__).resolve()

# Get project root directory
project_root = current_file.parent.parent

# Add project root to Python path
sys.path.append(str(project_root))

print(f"Project root: {project_root}")
print(f"Python path: {sys.path}")

import argparse

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data, load_data_0407
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist1(single_gpu=args.single_gpu)
    logger.configure(args.log_root)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data_0407(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        random_flip=args.random_flip,
        imablancedsample=args.imablancedsample,
    )

    logger.log("training...")
    
    # Load model weights and remove label embedding layer if exists
    checkpoint = torch.load(' ')
    if 'label_emb.weight' in checkpoint:
        del checkpoint['label_emb.weight']
    
    model.load_state_dict(checkpoint, strict=False)

    # Create new optimizer instead of loading old state
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-5,
        weight_decay=0.00001,
        lr_anneal_steps=0,
        batch_size=4,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=400,
        save_interval=20000,
        resume_checkpoint="",  # Initial model weight pth file path
        use_fp16=False,
        fp16_scale_growth=1e-3,
        log_root="./logs",
        imablancedsample=False,  # Sampling strategy for imbalanced dataset (single GPU only)
        random_flip=True,
        single_gpu=False,
        image_size=128,  # Image size
        in_channels=3,  # Number of input channels
        num_classes=13,  # Number of classes
        prob_uncon=0,  # Probability of classless embedding during training
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()