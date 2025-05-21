# coding=utf-8
import os
import re
import numpy as np
import tqdm
from PIL import Image
import argparse

# Color map for Potsdam dataset labels (RGB values)
# Each class has a specific color code for segmentation
Potsdam_COLOR_MAP = [
    [255, 0, 0],    # Clutter/background (Red)
    [255, 255, 255],# Impervious surfaces (White)
    [255, 255, 0],  # Car (Yellow)
    [0, 255, 0],    # Tree (Green)
    [0, 255, 255],  # Low vegetation (Cyan)
    [0, 0, 255]     # Building (Blue)
]

def RGB2Label(label, COLOR_MAP):
    """
    Convert RGB label image to class index mask
    Args:
        label: RGB label image (H, W, 3)
        COLOR_MAP: List of RGB values for each class
    Returns:
        temp_mask: Class index mask (H, W)
    """
    width, height = label.shape[0], label.shape[1]
    temp_mask = np.zeros(shape=(width, height))
    # Assign class index to each pixel based on color
    for index, color in enumerate(COLOR_MAP):
        # Find all pixels matching current color
        locations = np.all(label == color, axis=-1)
        temp_mask[locations] = index
    return temp_mask.astype(dtype=np.int8)

class Potsdam:
    """
    Class for processing the Potsdam remote sensing dataset
    Handles image splitting and label conversion
    """
    def __init__(self, dataset_path, target_path, image_type='RGB'):
        """
        Initialize dataset paths and file patterns
        Args:
            dataset_path: Path to source dataset
            target_path: Path to save processed data
            image_type: Type of image to process ('RGB' or 'IRRG')
        """
        # Regex pattern to extract file IDs (format: number_number)
        my_re = re.compile(r'\d+_\d+')
        self.dataset_path = dataset_path
        self.target_path = target_path
        self.image_type = image_type.upper()  # Normalize to uppercase
        
        # Validate image type
        if self.image_type not in ['RGB', 'IRRG']:
            raise ValueError("image_type must be either 'RGB' or 'IRRG'")
        
        # Path configuration (now supports both RGB and IRRG)
        self.image_path = os.path.join(dataset_path, self.image_type)
        self.Label_path = os.path.join(dataset_path, 'Label')  
        
        # Extract unique file identifiers from images
        self.file_flag = [
    	my_re.findall(name)[-1]
    	for name in os.listdir(self.image_path)
    	if name.endswith('.tif') and not re.search(r'4_12', name)]

    def start_dealWith(self, split_size):
        """
        Process dataset by splitting images and converting labels
        Args:
            split_size: Size (in pixels) for square image splits
        """
        # Create output directories if they don't exist
        os.makedirs(os.path.join(self.target_path, 'img_dir/train'), exist_ok=True)
        os.makedirs(os.path.join(self.target_path, 'ann_dir/train'), exist_ok=True)
        
        # Process each image with progress bar
        tqdm_file_flag = tqdm.tqdm(self.file_flag, total=len(self.file_flag))
        for flag in tqdm_file_flag:
            num = 0
            # Load corresponding label and RGB image
            label = np.array(Image.open(os.path.join(
                self.Label_path, 'top_potsdam_' + flag + '_label.tif')))
            img = np.array(Image.open(os.path.join(
                self.image_path, 'top_potsdam_' + flag + '_'+self.image_type+'.tif')))
            
            # Special case handling for file 6_7
            if flag == '6_7':
                label[label == 252] = 255  # Fix incorrect pixel value
            # Convert RGB label to class index mask
            mask = RGB2Label(label=label, COLOR_MAP=Potsdam_COLOR_MAP)
            
            # Determine split boundaries
            min_x = min(img.shape[0], mask.shape[0])
            min_y = min(img.shape[1], mask.shape[1])
            range_x = min_x // split_size
            range_y = min_y // split_size
            
            # Split images and save patches
            for x in range(range_x):
                for y in range(range_y):
                    # Extract image patch
                    split_image = img[
                        x * split_size:(x + 1) * split_size, 
                        y * split_size:(y + 1) * split_size]
                    
                    # Extract corresponding label patch
                    split_mask = mask[
                        x * split_size:(x + 1) * split_size, 
                        y * split_size:(y + 1) * split_size]
                    
                    # Save image and label patches
                    Image.fromarray(split_image).save(os.path.join(
                        self.target_path, 'img_dir/train',
                        'top_potsdam_' + flag + '_' + str(num) + '.png'))
                    Image.fromarray(split_mask).save(os.path.join(
                        self.target_path, 'ann_dir/train',
                        'top_potsdam_' + flag + '_' + str(num) + '.png'))
                    num += 1
        tqdm_file_flag.close()


def main():
    # Configure command line argument parser
    parser = argparse.ArgumentParser(description='Process Potsdam dataset.')
    parser.add_argument('--source', type=str, required=True, 
                       help='Path to the source dataset directory')
    parser.add_argument('--target', type=str, required=True,
                       help='Path to save processed files')
    parser.add_argument('--split_size', type=int, default=512,
                       help='Patch size for splitting (default: 512)')
    parser.add_argument('--image_type', type=str, default='RGB',
                       choices=['RGB', 'IRRG'],
                       help='Image type to process (RGB or IRRG)')

    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate source path exists
    if not os.path.exists(args.source):
        raise ValueError(f"Source path does not exist: {args.source}")
    
    # Initialize processor and start conversion
    p = Potsdam(args.source, args.target,args.image_type)
    print(f"Processing dataset from: {args.source}")
    print(f"Saving results to: {args.target}")
    print(f"Using split size: {args.split_size}")
    p.start_dealWith(split_size=args.split_size)

if __name__ == '__main__':
    main()
