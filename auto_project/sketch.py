import os

import cv2
from utils import process_folder

base_path = "./dataset/v2/img1.png"
sample_dir = "./dataset/v2"
output_dir = "./dataset/v2/diff_img"

def process_folder(base_img_path, sample_dir, output_dir):
    """
    For each .png image in sample_dir, subtract the base image,
    and save the result to output_dir with the same filename.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    base_img = cv2.imread(base_img_path)
    if base_img is None:
        print(f"Base image {base_img_path} not found or cannot be read.")
        return
    for filename in os.listdir(sample_dir):
        if not filename.lower().endswith('.png'):
            continue
        sample_path = os.path.join(sample_dir, filename)
        output_path = os.path.join(output_dir, filename)
        if not os.path.isfile(sample_path):
            continue
        sample_img = cv2.imread(sample_path)
        if sample_img is None:
            continue
        # Ensure images are the same size
        if sample_img.shape != base_img.shape:
            print(f"Shape mismatch for {filename}, skipping.")
            continue
        diff_img = cv2.subtract(sample_img, base_img)
        cv2.imwrite(output_path, diff_img)

process_folder(base_path, sample_dir, output_dir)
