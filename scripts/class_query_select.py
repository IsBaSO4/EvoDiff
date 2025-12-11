import pandas as pd
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from typing import Dict, Tuple, List, Optional, Set
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path
import os
from PIL import Image
import shutil
import csv
import os
import csv
import shutil
import math


def load_images_from_folder(folder_path: str) -> Tuple[List[torch.Tensor], List[str], List[Path]]:
    """
    Load all PNG images from a folder
    """
    images = []
    names = []
    paths = []
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                images.append(img_tensor)
                names.append(filename)
                paths.append(Path(img_path))
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
    
    if not images:
        raise ValueError(f"No PNG images found in {folder_path}")
        
    return images, names, paths


def load_images_from_merged_folders(base_folder: str, labels: List[str]) -> Tuple[List[torch.Tensor], List[str], List[Path]]:
    """
    Load and merge all PNG images from multiple subfolders
    """
    all_images = []
    all_names = []
    all_paths = []
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    for label in labels:
        label_folder = os.path.join(base_folder, label)
        if not os.path.exists(label_folder):
            print(f"Warning: Folder does not exist: {label_folder}")
            continue
            
        for filename in os.listdir(label_folder):
            if filename.lower().endswith('.png'):
                img_path = os.path.join(label_folder, filename)
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img)
                    all_images.append(img_tensor)
                    all_names.append(filename)
                    all_paths.append(Path(img_path))
                except Exception as e:
                    print(f"Error loading image {filename}: {e}")
    
    if not all_images:
        raise ValueError(f"No PNG images found in {base_folder} with labels {labels}")
        
    return all_images, all_names, all_paths


def load_all_images_from_folder(folder_path: str) -> torch.Tensor:
    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.png')])
    if not image_files:
        raise ValueError(f"No PNG images found in {folder_path}")
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    all_images = []
    for filename in image_files:
        img_path = os.path.join(folder_path, filename)
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            all_images.append(img_tensor)
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
    
    if all_images:
        return torch.stack(all_images)
    else:
        raise ValueError(f"No valid images loaded from {folder_path}")


def load_all_images_from_merged_folders(base_folder: str, labels: List[str]) -> torch.Tensor:
    all_images = []
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    for label in labels:
        label_folder = os.path.join(base_folder, label)
        if not os.path.exists(label_folder):
            print(f"Warning: Folder does not exist: {label_folder}")
            continue
            
        # Load and sort all images in this label folder
        image_files = sorted([f for f in os.listdir(label_folder) if f.lower().endswith('.png')])
        
        for filename in image_files:
            img_path = os.path.join(label_folder, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                all_images.append(img_tensor)
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
    
    if all_images:
        return torch.stack(all_images)
    else:
        raise ValueError(f"No valid images loaded from {base_folder} with labels {labels}")


def get_saved_image_paths(output_folders: List[str]) -> Set[str]:
    """
    Get the set of saved image paths
    """
    saved_paths = set()
    
    for folder in output_folders:
        if not os.path.exists(folder):
            continue
            
        # Check 0 and 1 subfolders directly
        for label in ["0", "1"]:
            label_folder = os.path.join(folder, label)
            if os.path.exists(label_folder):
                for filename in os.listdir(label_folder):
                    if filename.lower().endswith('.png'):
                        # Use filename only for simplification
                        saved_paths.add(filename)
    
    return saved_paths


@dataclass
class DataItem:
    """Data item class for storing image information and similarities"""
    image_name: str                    # Image name
    image_path: Path                   # Image path
    similarities: Dict[str, float]     # Similarities with images {batch_name: similarity}
    avg_similarity: float = 0.0        # Average similarity
    selected: bool = False             # Whether selected
    
    def update_avg_similarity(self):
        """Update average similarity"""
        self.avg_similarity = np.mean(list(self.similarities.values()))
        
    def __str__(self):
        """String representation"""
        return (f"Image: {self.image_name}, "
                f"Avg Similarity: {self.avg_similarity:.4f}, "
                f"Selected: {self.selected}")


# ResNet50 pretrained weights path
init_weight = ' '


class FeatureExtractor:
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        # Load pretrained ResNet model
        self.model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        checkpoint = torch.load(os.path.join(init_weight, 'best_model.pth'))
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        # Remove final fully connected layer, use feature extraction part only
        self.model = self.model.to(device)
        self.model.eval()

        # Define image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((128,128))
        ])
        
    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract image features
        """
        # Ensure images are on correct device
        images = images.to(self.device)
        
        # Normalize images
        images = self.transform(images)
        
        # Extract features
        features = self.model(images)
        features = features.squeeze()
        
        # L2 normalize features
        features = F.normalize(features, p=2, dim=1)
        
        return features


class BatchQuerySelector:
    def __init__(self, selection_percentage: float = 0.05):
        """
        Initialize query selector
        
        Args:
            selection_percentage (float): Selection percentage (default: 5%)
        """
        self.selection_percentage = selection_percentage
        self.feature_extractor = FeatureExtractor()
        self.data_items: List[DataItem] = []
       
    def compute_image_similarities(self, 
                                 unlabeled_images: torch.Tensor,
                                 batch_images: torch.Tensor) -> torch.Tensor:
        # Extract features from unlabeled images
        unlabeled_features = self.feature_extractor.extract_features(unlabeled_images)
        
        # Extract features from batch images
        batch_features = self.feature_extractor.extract_features(batch_images)
        
        # Calculate cosine similarity
        similarities = torch.mm(unlabeled_features, batch_features.t())
        
        return similarities

    def select_samples(self, similarities: torch.Tensor) -> torch.Tensor:
        n_samples = similarities.shape[0]
        n_select = int(n_samples * self.selection_percentage)
        
        # Calculate maximum similarity for each unlabeled sample
        max_similarities, _ = torch.max(similarities, dim=1)
        
        # Select samples with highest similarity
        _, selected_indices = torch.topk(max_similarities, k=n_select)
        
        return selected_indices

    def process_images(self, 
                      unlabeled_images: torch.Tensor,
                      unlabeled_names: List[str],
                      unlabeled_paths: List[Path],
                      batch_images: torch.Tensor) -> List[DataItem]:
        # Compute similarities
        similarities = self.compute_image_similarities(unlabeled_images, batch_images)
        
        # Create DataItem list
        self.data_items = []
        for idx, (name, path) in enumerate(zip(unlabeled_names, unlabeled_paths)):
            # Get maximum similarity for current image
            max_sim = similarities[idx].max().item()
            
            # Create DataItem object
            data_item = DataItem(
                image_name=name,
                image_path=path,
                similarities={"all_images": max_sim}
            )
            data_item.update_avg_similarity()
            self.data_items.append(data_item)
            
        return self.data_items
    
    def select_top_samples(self) -> List[DataItem]:
        # Ensure data_items have been created
        if not self.data_items:
            raise ValueError("No data items available. Please run process_images first.")
        
        # Calculate number of samples to select
        n_select = int(len(self.data_items) * self.selection_percentage)
        
        # Sort by average similarity
        sorted_items = sorted(self.data_items, 
                            key=lambda x: x.avg_similarity, 
                            reverse=True)
        
        # Mark selected samples
        selected_items = []
        for item in sorted_items[:n_select]:
            item.selected = True
            selected_items.append(item)
            
        return selected_items
    
    def get_selected_image_paths(self) -> List[Path]:
        return [item.image_path for item in self.data_items if item.selected]
    
    def save_results(self, output_file: str):
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Image Name", "Average Similarity", "Selected", "Batch Similarities"])
            
            for item in self.data_items:
                batch_sims = ';'.join([f"{k}:{v:.4f}"
                                     for k, v in item.similarities.items()])
                writer.writerow([
                    str(item.image_path),
                    f"{item.avg_similarity:.4f}",
                    str(item.selected),
                    batch_sims
                ])


def process_single_scale_merged(unlabeled_base: str, generated_base: str, scale_name: str, labels: List[str], base_output_path: str):

    print(f"\nProcessing {scale_name} - Merging all labels {labels}...")
    
    # Load unlabeled images (merge all labels)
    unlabeled_images_list, unlabeled_names, unlabeled_paths = load_images_from_merged_folders(unlabeled_base, labels)
    unlabeled_images = torch.stack(unlabeled_images_list)
    
    # Load generated images (merge all labels)
    generated_folder = os.path.join(generated_base, scale_name)
    batch_images = load_all_images_from_merged_folders(generated_folder, labels)
    
    # Initialize selector
    selector = BatchQuerySelector(selection_percentage=0.1)
    
    # Process images and create DataItem list
    data_items = selector.process_images(
        unlabeled_images, 
        unlabeled_names,
        unlabeled_paths,
        batch_images
    )
    
    # Select samples
    selected_items = selector.select_top_samples()
    
    # Create output path
    csv_path = os.path.join(base_output_path, f"{scale_name}_all.csv")
    
    # Save results
    selector.save_results(csv_path)
    
    # Print results
    print(f"Total number of images: {len(data_items)}")
    print(f"Number of selected images: {len(selected_items)}")
    print(f"Results saved to {csv_path}")
    
    return csv_path


def save_selected_images_no_duplicate(csv_path: str, output_base_folder: str, ratio: float, excluded_images: Set[str]):
    # Create 0 and 1 subfolders directly in output folder
    os.makedirs(os.path.join(output_base_folder, '0'), exist_ok=True)
    os.makedirs(os.path.join(output_base_folder, '1'), exist_ok=True)   
    
    # Record number of copied images and filenames
    copied_count = {'0': 0, '1': 0}
    copied_images = set()

    try:
        # Read CSV file and sort by similarity (should already be sorted)
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header row
            
            all_rows = list(reader)
        
        # Sort by similarity (although should already be sorted)
        all_rows.sort(key=lambda x: float(x[1]), reverse=True)
        
        # Calculate total number of images needed
        total_needed = math.ceil(len(all_rows) * ratio)
        current_copied = 0
        
        print(f"Need to copy {total_needed} images, excluding {len(excluded_images)} existing images")
        
        # Iterate through all rows until enough images are copied
        for row in all_rows:
            if current_copied >= total_needed:
                break
                
            # Get image path
            image_path = row[0]
            image_filename = os.path.basename(image_path)
            
            # Check if duplicate image
            if image_filename in excluded_images:
                print(f"Skipping duplicate image: {image_filename}")
                continue
            
            # Extract label from path (second to last directory name)
            path_parts = image_path.split(os.sep)
            original_label = path_parts[-2]
            
            # Map original label to output label
            if original_label == '0':
                output_label = '0'
            elif original_label == '1':
                output_label = '1'
            else:
                output_label = original_label
            
            if output_label in ['0', '1']:
                # Build destination path - save directly to output_base_folder/label/
                dest_path = os.path.join(output_base_folder, output_label, image_filename)
                
                try:
                    # Copy image to new folder
                    shutil.copy2(image_path, dest_path)
                    copied_count[output_label] += 1
                    copied_images.add(image_filename)
                    current_copied += 1
                    print(f"Copied ({ratio*100}%): {image_path} -> {dest_path}")
                except Exception as e:
                    print(f"Error copying {image_path}: {str(e)}")

        print(f"\n{ratio*100}% ratio processing complete (no duplicates):")
        print(f"Label 0: copied {copied_count['0']} images")
        print(f"Label 1: copied {copied_count['1']} images")
        print(f"Total copied {current_copied} images")
        print(f"Skipped {len(excluded_images)} duplicate images")
        print("-" * 50)

        return copied_images

    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return set()


def main():
    # Base path configuration
    unlabeled_base = " "
    generated_base = " "
    output_base = " "
    
    # Scale configuration - process in specified order
    scales = ["scale1", "scale3", "scale5", "scale7", "scale9"]
    labels = ["0", "1"]  # Original labels
    ratio = 0.1  # default ratio at 0.1
    
    # Create base output directory
    os.makedirs(output_base, exist_ok=True)
    
    # Track saved images
    all_saved_images = set()
    processed_folders = []
    
    # Process each scale in order
    for scale_idx, scale in enumerate(scales):
        print(f"\n{'='*50}")
        print(f"Starting to process {scale} ({scale_idx+1}/{len(scales)})")
        print(f"Merging labels: {labels}")
        print(f"{'='*50}")
        
        # Check if folders exist
        generated_folder = os.path.join(generated_base, scale)
        all_labels_exist = True
        
        for label in labels:
            unlabeled_folder = os.path.join(unlabeled_base, label)
            generated_subfolder = os.path.join(generated_folder, label)
            
            if not os.path.exists(unlabeled_folder):
                print(f"Warning: Unlabeled folder does not exist: {unlabeled_folder}")
                all_labels_exist = False
                
            if not os.path.exists(generated_subfolder):
                print(f"Warning: Generated folder does not exist: {generated_subfolder}")
                all_labels_exist = False
        
        if not all_labels_exist:
            print(f"Skipping {scale} because some required folders do not exist")
            continue
        
        # Process single scale (merge all labels)
        csv_path = process_single_scale_merged(
            unlabeled_base, 
            generated_base, 
            scale, 
            labels, 
            output_base
        )
        
        # Create output path
        folder_output = os.path.join(output_base, f"choose_cosine_{scale}_all")
        
        # Use no-duplicate save function
        print(f"\nStarting to save selected images for {scale} (avoiding duplicates with previous scales)...")
        newly_saved = save_selected_images_no_duplicate(
            csv_path, 
            folder_output, 
            ratio, 
            all_saved_images
        )
        
        # Update saved images set
        all_saved_images.update(newly_saved)
        processed_folders.append(folder_output)
        
        print(f"\n{scale} processing complete:")
        print(f"CSV file: {csv_path}")
        print(f"Output folder: {folder_output}")
        print(f"New images count: {len(newly_saved)}")
        print(f"Total saved images count: {len(all_saved_images)}")
    
    print(f"\n{'='*50}")
    print("All scales processing complete!")
    print(f"{'='*50}")
    
    # Print output summary
    print("\nOutput files summary:")
    for scale in scales:
        output_file = os.path.join(output_base, f"{scale}_all.csv")
        folder_output = os.path.join(output_base, f"choose_cosine_{scale}_all")
        if os.path.exists(output_file):
            print(f"- {scale}: CSV file -> {output_file}")
            print(f"  {scale}: Image folder -> {folder_output}")
    
    print(f"\nTotal saved {len(all_saved_images)} unique images")
    print("Processing order: scale1 → scale3 → scale5 → scale7 → scale9")
    print("Each subsequent scale avoids images already saved in previous scales")
    print("Images are saved directly in 0 and 1 subdirectories of each scale folder")
    print("All label images are merged for similarity calculation")


if __name__ == "__main__":
    main()