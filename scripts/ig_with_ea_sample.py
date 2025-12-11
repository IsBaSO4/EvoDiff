import os
import sys
from pathlib import Path
import torch

# Get absolute path of current file
current_file = Path(__file__).resolve()

# Get project root directory
project_root = current_file.parent.parent

# Add project root to Python path
sys.path.append(str(project_root))

# Print debug information
print(f"Project root: {project_root}")
print(f"Python path: {sys.path}")

from config import settings

import cv2
import argparse
import os
from PIL import Image
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.utils as tvu

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()
    
    dist_util.setup_dist1(args.single_gpu)
    logger.configure(args.log_root)

    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Current device: {dist_util.dev()}")
    print(f"World size (number of processes): {dist.get_world_size()}")
    print(f"Current rank: {dist.get_rank()}")

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # Create and load classifier
    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    if args.classifier_path != "":
        classifier.load_state_dict(
            dist_util.load_state_dict(args.classifier_path, map_location="cpu")
        )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def LOEA(x_in, t):
        """
        Local Optimization Evolutionary Algorithm (LOEA)
        Searches for solutions in parameter space that maximize classifier output entropy
        using evolutionary algorithms.
        
        Args:
            x_in: Initial population with shape [pop_size, ...]
                  Each individual in the population is an input to be optimized
        
        Returns:
            x_loea: Individual with maximum entropy after evolution, maintaining batch dimension [1, ...]
        """
        pop = x_in  # Initial population: [pop_size, ...]
        num_generations = 40  # Number of evolution generations
        pop_size = pop.shape[0]
        mutation_prob = 0.1  # Mutation probability
        mutation_std = 0.1   # Mutation noise standard deviation

        for gen in range(num_generations):
            # 1. Calculate fitness (entropy) for each individual in population
            logits = classifier(pop, t)
            log_probs = F.log_softmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)
            # Calculate Shannon entropy for each individual (sum over class dimension)
            # Higher entropy indicates more uncertain model predictions
            fitness = -(probs * log_probs).sum(dim=1)  # shape: [pop_size]

            # 2. Select top 50% individuals with highest fitness
            num_selected = pop_size // 2
            _, top_idxs = fitness.topk(num_selected, largest=True)
            selected = pop[top_idxs]

            # 3. Crossover: randomly pair selected individuals to generate offspring
            offspring = []
            for i in range(pop_size):
                # Randomly select two parents
                idx1 = th.randint(0, num_selected, (1,)).item()
                idx2 = th.randint(0, num_selected, (1,)).item()
                parent1 = selected[idx1]
                parent2 = selected[idx2]
                # Element-wise random selection for crossover
                mask = th.randint(0, 2, parent1.shape, dtype=parent1.dtype, device=parent1.device)
                child = mask * parent1 + (1 - mask) * parent2
                offspring.append(child.unsqueeze(0))
            offspring = th.cat(offspring, dim=0)  # New population: [pop_size, ...]

            # 4. Mutation: add noise to offspring with probability mutation_prob
            mutation_mask = th.rand_like(offspring) < mutation_prob
            noise = th.randn_like(offspring) * mutation_std
            offspring = offspring + mutation_mask * noise

            # Update population
            pop = offspring

        # After evolution, select individual with maximum entropy as final result
        logits = classifier(pop, t)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        fitness = -(probs * log_probs).sum(dim=1)
        best_idx = fitness.argmax()
        x_loea = pop[best_idx:best_idx+1]  # Maintain batch dimension
        return x_loea
        
    def cond_fn(x, t, y=None):
        """
        Conditional guidance function for classifier-guided diffusion
        
        Args:
            x: Input image tensor
            t: Timestep
            y: Label (must be provided)
        
        Returns:
            Gradient for guided sampling
        """
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)
            
            # Calculate original entropy
            entropy_original = -(probs * log_probs).sum()
                
            # Apply LOEA to input
            x_loea = LOEA(x_in, t)
            
            # Calculate entropy after LOEA
            logits_loea = classifier(x_loea, t)
            log_probs_loea = F.log_softmax(logits_loea, dim=-1)
            probs_loea = F.softmax(logits_loea, dim=-1)
            
            # Information gain: maximize entropy difference
            entropy_loea = -(probs_loea * log_probs_loea).sum()
            information_gain = entropy_loea - entropy_original
            
            # Combined objective: minimize information gain while maximizing original entropy
            combined_objective = args.classifier_scale * information_gain
            
            # Compute gradient with respect to x_in
            grad_combined = th.autograd.grad(combined_objective, x_in)[0]
            return grad_combined

    def model_fn(x, t, y=None):
        """Wrapper function for model forward pass"""
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    num_samples = 0

    # Calculate total samples and per-class quantities
    for ele in range(0, len(args.category_num_list)):
        num_samples = num_samples + args.category_num_list[ele]
    lis = []
    for i in range(len(args.category_num_list)):
        lis.extend([i] * args.category_num_list[i])

    # Generate images in batches
    while len(all_images) * args.batch_size < num_samples:
        model_kwargs = {}
        
        # Get class labels for current batch
        classes = th.tensor(
            lis[len(all_images) * args.batch_size:len(all_images) * args.batch_size + args.batch_size],
            device=dist_util.dev()
        )
        print(classes)

        model_kwargs["y"] = classes
        
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        
        # Generate samples
        sample = sample_fn(
            model_fn,
            (len(classes), 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=None if not args.classifier_scale else cond_fn,
            device=dist_util.dev(),
            cfg=args.cfg
        )
        
        if args.get_image:
            sample = ((sample + 1) / 2).clamp(0, 1)
            tvu.save_image(sample, os.path.join(logger.get_dir(), f"output.png"))
            sample = sample.to(th.uint8)
            break
        else:
            # Convert from [-1, 1] to [0, 255]
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)

        # Reshape: (batch, channel, height, width) -> (batch, height, width, channel)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        # Distributed gathering
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, classes)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    # Concatenate all batches and truncate to desired number of samples
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: num_samples]
    
    # Save images (only on main process)
    if dist.get_rank() == 0:
        if args.get_images:
            # Create directory structure
            if not os.path.exists(os.path.join(args.dataset_dir, args.dataset_name)):
                os.makedirs(os.path.join(args.dataset_dir, args.dataset_name))
                for i in args.category_name_list:
                    os.makedirs(os.path.join(args.dataset_dir, args.dataset_name, i))
            
            # Save individual image files by category
            for i in range(len(arr)):
                im = Image.fromarray(arr[i])
                im.save(os.path.join(
                    args.dataset_dir,
                    args.dataset_name,
                    args.category_name_list[label_arr[i]],
                    str(i) + ".png"
                ))
        else:
            print('0')

    dist.barrier()
    logger.log("sampling complete")
    sys.exit(0)


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        batch_size=settings.SAMPLE_BATCH_SIZE,
        use_ddim=True,
        model_path=settings.SAMPLE_MODEL_PATH,
        classifier_path=settings.SAMPLE_CLASSIFIER_PATH,
        classifier_scale=settings.SAMPLE_CLASSIFIER_SCALE_00000,
        log_root=settings.SAMPLE_LOG_ROOT,
        dataset_dir=settings.SAMPLE_DATASET_DIR,
        category_name_list=settings.SAMPLE_CATEGORY_NAME_LIST,
        category_num_list=settings.SAMPLE_CATEGORY_NUM_LIST,
        dataset_name=settings.SAMPLE_DATASET_NAME_00000,
        get_image=False,
        get_images=True,
        single_gpu=False,
        cfg=settings.CFG,
        image_size=128,      # Image size
        in_channels=3,       # Number of input channels
        num_classes=13,      # Number of classes
        prob_uncon=0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def setup_seed(seed):
    """Set random seed for reproducibility"""
    import random
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    th.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # Set random seed for reproducibility
    setup_seed(42)
    main()