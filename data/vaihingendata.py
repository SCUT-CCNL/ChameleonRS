import os
import argparse
import numpy as np
from PIL import Image
import tqdm

# Vaihingen dataset RGB to class index mapping (example - adjust based on your labels)
Vaihingen_COLOR_MAP = {
    (255, 0, 0): 0,       # Clutter
    (255, 255, 255): 1,  # Impervious surfaces
    (255, 255, 0): 2,    # Cars
    (0, 255, 0): 3,      # Trees
    (0, 255, 255): 4,    # Low vegetation
    (0, 0, 255): 5      # Buildings
    
}

def RGB2Label(label_img, color_map):
    """Convert RGB label image to class index mask.
    Args:
        label_img: (H,W,3) RGB numpy array
        color_map: Dict mapping RGB tuples to class indices
    Returns:
        (H,W) numpy array with class indices
    """
    h, w = label_img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for rgb, idx in color_map.items():
        match_pixels = (label_img == np.array(rgb)).all(axis=-1)
        mask[match_pixels] = idx
    
    return mask

class VaihingenProcessor:
    def __init__(self, source_root, target_root, mode):
        """Initialize processor for Vaihingen dataset.
        Args:
            source_root: Path to raw dataset (contains images/, labels/)
            target_root: Output directory for processed data
            mode: One of ['train', 'val', 'test'] (determines which files to load)
        """
        self.source_dir = source_root
        self.target_img_dir = os.path.join(target_root, 'img_dir', mode)
        self.target_ann_dir = os.path.join(target_root, 'ann_dir', mode)
        
        # Paths to original data
        self.image_dir = os.path.join(source_root, 'images')
        self.label_dir = os.path.join(source_root, 'labels')
        
        # Validate directories
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory missing: {self.image_dir}")
        if not os.path.exists(self.label_dir):
            raise FileNotFoundError(f"Label directory missing: {self.label_dir}")
        
        # Load file list for current mode (train/val/test)
        self.file_list = self._load_file_list(mode)
        
        # Create output directories
        os.makedirs(self.target_img_dir, exist_ok=True)
        os.makedirs(self.target_ann_dir, exist_ok=True)

    def _load_file_list(self, mode):
        """Load filenames from {mode}.txt (e.g., train.txt).
        Returns:
            List of filenames (without extensions)
        """
        txt_path = os.path.join(self.source_dir, f"{mode}.txt")
        with open(txt_path, 'r') as f:
            files = [line.strip() for line in f if line.strip()]
        return files

    def process(self, split_size=512):
        """Process images and labels:
        1. Load image/label pairs
        2. Convert labels to class indices
        3. Split into patches
        4. Save patches to target directories
        """
        for filename in tqdm.tqdm(self.file_list, desc=f"Processing {len(self.file_list)} files"):
            # Build full paths (adjust extensions if needed)
            img_path = os.path.join(self.image_dir, f"{filename}.tif")
            label_path = os.path.join(self.label_dir, f"{filename}.tif")
            
            # Skip if files are missing
            if not os.path.exists(img_path):
                print(f"Warning: Image {filename} missing!")
                continue
            if not os.path.exists(label_path):
                print(f"Warning: Label {filename} missing!")
                continue
            
            try:
                # Load and validate image/label
                image = np.array(Image.open(img_path))
                label = np.array(Image.open(label_path))
                
                # Convert RGB label to class indices
                mask = RGB2Label(label, Vaihingen_COLOR_MAP)
                
                # Split into patches (implementation depends on your needs)
                self._save_patches(image, mask, filename, split_size)
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    def _save_patches(self, image, mask, base_name, patch_size, stride=256):
        """Split and save image/label patches.
        Args:
            image: (H,W,3) RGB numpy array
            mask: (H,W) class index mask
            base_name: Base filename for patches
            
        """
        h, w = image.shape[:2]
        count = 0
        
        for y in range(0, h - patch_size + 1, stride):  
            for x in range(0, w - patch_size + 1, stride):
                img_patch = image[y : y + patch_size, x : x + patch_size]
                mask_patch = mask[y : y + patch_size, x : x + patch_size]
                
                patch_name = f"{base_name}_{count}"
                Image.fromarray(img_patch).save(os.path.join(self.target_img_dir, f"{patch_name}.png"))
                Image.fromarray(mask_patch).save(os.path.join(self.target_ann_dir, f"{patch_name}.png"))
                count += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Path to raw Vaihingen dataset')
    parser.add_argument('--target', type=str, required=True, help='Output directory for processed data')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'val', 'test'], 
                       help='Processing mode')
    parser.add_argument('--patch_size', type=int, default=512, help='Size of image patches')
    args = parser.parse_args()
    
    processor = VaihingenProcessor(args.source, args.target, args.mode)
    processor.process(args.patch_size)

if __name__ == '__main__':
    main()
