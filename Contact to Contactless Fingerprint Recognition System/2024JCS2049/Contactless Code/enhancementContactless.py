import cv2 as cv
import os
import numpy as np
from tqdm import tqdm
from normalization import normalize

def invert_image(image):
    return cv.bitwise_not(image)  

def enhancement(image):
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
    enhanced_img = clahe.apply(image)
    img = invert_image(enhanced_img)
    return img

input_dir = './segmentedOutput'
output_dir = './preProcessedContactLessImages'


image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')  
file_paths = []
for root, _, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith(image_extensions):
            file_paths.append(os.path.join(root, file))


for file_path in tqdm(file_paths, desc="Enhancing images"):
    
    image = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Warning: Could not read {file_path}. Skipping...")
        continue

    result = enhancement(image)
    
    normalizedImage = normalize(result.copy(), float(60), float(60))

    relative_folder = os.path.relpath(os.path.dirname(file_path), input_dir)
    output_folder = os.path.join(output_dir, relative_folder)
    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, os.path.basename(file_path))
    cv.imwrite(output_path, normalizedImage)
