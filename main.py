from networks.transforms import trimap_transform, normalise_image
from networks.models import build_model
from dataloader import PredDataset
from networks.isnet import ISNetDIS

# System libs
import os
import argparse

# External libs
import cv2
import numpy as np
import torch
import time
from tqdm import tqdm
from utils import find_all_images, check_output_exists, create_binary_mask, create_output_with_black_background, create_output_with_white_background, save_outputs
from utils import load_isnet_model
from config import *
from utils import process_single_image


FBA_MODEL_DIR = '../model_weights'



def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Background Removal Inference Script')
    parser.add_argument('--gsync', action='store_true', 
                       help='Sync output folders to GCP bucket after processing')
    parser.add_argument('--ckpt-dir-isnet', type=str, default=None,
                       help='Directory containing checkpoints (overrides config MODEL_SAVE_DIR)')
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Find all images in the input dataset
    print(f"Searching for images in: {INPUT_DATASET_PATH}")
    all_images = find_all_images(INPUT_DATASET_PATH, IMAGE_EXTENSIONS)
    
    print(f"Found {len(all_images)} images to process")

    if len(all_images) == 0:
        print("No images found! Please check the input path and extensions.")
        return

    ckpt_dir = args.ckpt_dir_isnet if args.ckpt_dir_isnet else MODEL_SAVE_DIR
    os.makedirs(INFERENCE_OUTPUT_DIR, exist_ok=True)

    # ----------------------
    # Load ISNet model
    # ----------------------
    isnet_model = None
    model_path = os.path.join(ckpt_dir, 'model.pth')
    if os.path.exists(model_path):
        try:
            isnet_model = load_isnet_model(model_path)
        except Exception as e:
            print(f"[ERROR] Error loading ISNet model: {e}")
            return
    else:
        print(f"[ERROR] ISNet checkpoint not found at {model_path}")
        return

    # ----------------------
    # Load FBA model (optional)
    # ----------------------
    fba_model = None
    fba_model_path = os.path.join(FBA_MODEL_DIR, 'FBA.pth')
    if os.path.exists(fba_model_path):
        try:
            fba_model = build_model(fba_model_path)
            if torch.cuda.is_available():
                fba_model = fba_model.cuda()
                print(f"FBA Model loaded on GPU: {fba_model_path}")
            else:
                print(f"FBA Model loaded on CPU: {fba_model_path}")
            fba_model.eval()
        except Exception as e:
            print(f"[WARN] Could not load FBA model: {e}")

    # ----------------------
    # Process images
    # ----------------------
    processed_count, skipped_count = 0, 0
    total_time = 0.0

    for image_path in tqdm(all_images):
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        if check_output_exists(INFERENCE_OUTPUT_DIR, image_name):
            skipped_count += 1
            continue

        try:
            start_time = time.time()
            success = process_single_image(isnet_model, fba_model, image_path, INFERENCE_OUTPUT_DIR, THRESHOLD)
            elapsed = time.time() - start_time

            if success:
                processed_count += 1
                total_time += elapsed
                print(f"[INFO] Processed {image_name} in {elapsed:.2f} sec")

        except Exception as e:
            print(f"[ERROR] Failed processing {image_name}: {e}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ----------------------
    # Summary
    # ----------------------
    print("\n===== SUMMARY =====")
    print(f"Processed images: {processed_count}")
    print(f"Skipped images : {skipped_count}")
    if processed_count > 0:
        print(f"Average time per image: {total_time / processed_count:.2f} sec")


if __name__ == '__main__':
    main()
    