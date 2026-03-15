# %% Imports
import torch
import torch.nn.functional as F
import argparse
import random
import pandas as pd
import csv
from transformers import SiglipModel, SiglipProcessor
from PIL import Image
import numpy as np
import os
from tqdm import tqdm # For progress bars
import gc
from sklearn.model_selection import train_test_split
import json


# --- Device Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# (Paste load_utkface_dataset and generate_embeddings functions here)
# ... (Assuming they are pasted) ...

def load_dataset(task_type, data_dir, dataset_name):
    img_dir = os.path.join(data_dir, "images")
    dataset = []

    print(f"Loading {dataset_name} data from: {data_dir}")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"{dataset_name} directory not found: {data_dir}")
    
    # load label files if needed
    if task_type == "aesthetic_score":
        ava_grade_dict = {}
        ava_txt_path = os.path.join(data_dir, "AVA_Files", "AVA.txt")
        with open(ava_txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                parts = [int(p) for p in parts]
                judger_num = sum(parts[2:12])
                avg_score = sum(parts[i] * (i - 1) for i in range(2, 12)) / judger_num
                ava_grade_dict[parts[1]] = avg_score
    elif task_type == "facial_beauty":
        rating_xlsx_path = os.path.join(data_dir, "All_Ratings.xlsx")
        df = pd.read_excel(rating_xlsx_path)
        facial_beauty_dict = {}
        for index, row in df.iterrows():
            filename = row['Filename']
            score = row['Rating']
            if filename not in facial_beauty_dict:
                facial_beauty_dict[filename] = [score]
            else:
                facial_beauty_dict[filename].append(score)
        for key in facial_beauty_dict:
            facial_beauty_dict[key] = sum(facial_beauty_dict[key]) / len(facial_beauty_dict[key])
    elif task_type == "modified_image_quality":
        quality_dict = {}
        quality_csv_path = os.path.join(data_dir, "dmos.csv")
        with open(quality_csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_name = row["dist_img"]
                dmos_score = float(row["dmos"])
                quality_dict[image_name] = dmos_score
    elif task_type == "wild_image_quality":
        quality_dict = {}
        quality_csv_path = os.path.join(data_dir, "koniq10k_scores_and_distributions.csv")
        with open(quality_csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_name = row["image_name"]
                mos_score = float(row["MOS"])
                quality_dict[image_name] = mos_score

    # find images
    for filename in sorted(os.listdir(img_dir)):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(img_dir, filename)
            if task_type == "age_prediction":
                try:
                    parts = filename.split('_')
                    age = int(parts[0])
                    if age < 0 or age > 116: continue
                    dataset.append((img_path, age))
                except (ValueError, IndexError):
                    continue
            
            elif task_type == "aesthetic_score":
                image_num = int(filename.split('.')[0])
                dataset.append((img_path, ava_grade_dict[image_num]))
            elif task_type == "facial_beauty":
                if filename in facial_beauty_dict:
                    dataset.append((img_path, facial_beauty_dict[f"{filename}"]))
            elif task_type == "modified_image_quality":
                if filename in quality_dict:
                    dataset.append((img_path, quality_dict[f"{filename}"]))
            elif task_type == "wild_image_quality":
                if filename in quality_dict:
                    dataset.append((img_path, quality_dict[f"{filename}"]))
            
    print(f"Loaded {len(dataset)} valid samples from {dataset_name}.")
    if len(dataset) == 0: raise ValueError(f"No valid {dataset_name} images found.")
    return dataset[:30]


def generate_embeddings(dataset, vision_model, processor):
    """Generates embeddings for a given dataset split."""
    print(f"Generating embeddings for {len(dataset)} images...")
    
    N_local = len(dataset)
    D_local = vision_model.config.vision_config.hidden_size # 768 for base model
    local_embeddings = torch.zeros((N_local, D_local), device='cpu', dtype=torch.float32) # Store on CPU first

    with torch.no_grad():
        batch_size = 128 
        for i in tqdm(range(0, N_local, batch_size), desc="Generating Embeddings"):
            batch_paths = [item[0] for item in dataset[i:min(i+batch_size, N_local)]]
            try:
                images = [Image.open(p).convert("RGB") for p in batch_paths]
                inputs = processor(images=images, return_tensors="pt").to(device)
                outputs = vision_model.get_image_features(**inputs)
                image_features = outputs.pooler_output
                local_embeddings[i:min(i+batch_size, N_local)] = image_features.cpu()
            except Exception as e:
                print(f"Error processing batch starting at index {i}: {e}")
                continue # Skip batch on error

    # Normalize
    local_embeddings = F.normalize(local_embeddings, p=2, dim=1)
    print(f"Normalized local_embeddings shape: {local_embeddings.shape}")
    print(f"Generated {local_embeddings.shape[0]} embeddings (on CPU).")
    
    # Explicitly delete models no longer needed
    torch.cuda.empty_cache()
    gc.collect()
    return local_embeddings


def main(args):
    dataset_name = os.path.basename(args.data_dir)
    
    print(f"\n=== Preparing data for task: {args.task_type} ({dataset_name}) ===\n")
    
    # --- 1. Load Full Dataset ---
    full_dataset = load_dataset(args.task_type, args.data_dir, dataset_name)
    print(f"Total dataset size: {len(full_dataset)}")
    
    # 2. Split the dataset
    print(f"Splitting dataset into Train ({1.0 - args.test_split:.0%}) and Test ({args.test_split:.0%})...")
    train_set, test_set = train_test_split(full_dataset, test_size=args.test_split, random_state=args.random_state)

    print(f"Train set size: {len(train_set)}")
    print(f"Test set size:  {len(test_set)}")

    # 3. Generate Embeddings for TRAIN set
    vision_model = SiglipModel.from_pretrained(args.vision_model_name).to(device).eval()
    processor = SiglipProcessor.from_pretrained(args.vision_model_name)

    print("\n--- Generating Training Embeddings ---")
    train_embeddings = generate_embeddings(train_set, vision_model, processor)

    # 4. Generate Embeddings for TEST set
    print("\n--- Generating Test Embeddings ---")
    test_embeddings = generate_embeddings(test_set, vision_model, processor)

    # 5. Save all files
    print("\n--- Saving all files ---")
    
    # Define output paths
    base_path = args.output_dir
    os.makedirs(base_path, exist_ok=True)
    train_dataset_file = os.path.join(base_path, f"train_dataset.pt")
    train_embedding_file = os.path.join(base_path, f"train_embeddings.pt")
    test_dataset_file = os.path.join(base_path, f"test_dataset.pt")
    test_embedding_file = os.path.join(base_path, f"test_embeddings.pt")
    
    # Save Train files
    # torch.save(train_set, train_dataset_file)
    # torch.save(train_embeddings, train_embedding_file)
    print(f"Saved: {train_dataset_file}")
    print(f"Saved: {train_embedding_file}")

    # Save Test files
    # torch.save(test_set, test_dataset_file)
    # torch.save(test_embeddings, test_embedding_file)
    print(f"Saved: {test_dataset_file}")
    print(f"Saved: {test_embedding_file}")

    print("\n--- Data preparation complete! ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Siglip embeddings for dataset.")
    parser.add_argument("--task_type", type=str, required=True, help="Type of task for dataset preparation.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory.")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Where to save .pt files.")
    parser.add_argument("--vision_model_name", type=str, default="google/siglip-base-patch16-224", help="Vision model name.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for dataset split.")
    parser.add_argument("--test_split", type=float, default=0.2, help="Fraction of data for test set.")
    args = parser.parse_args()
    

    main(args)