import os
from PIL import Image
import random
import glob

class EnhanceRedTransform(object):
    def __init__(self, enhanced_red_factor=4.5):
        self.enhanced_red_factor = enhanced_red_factor

    def __call__(self, img):
        r, g, b = img.split()
        enhanced_red = r.point(lambda i: i * self.enhanced_red_factor)
        return Image.merge('RGB', (enhanced_red, g, b))

# Assuming you have a directory with images
source_dir = '/data/dhruv_gautam/flickr30k_images/flickr30k_images/tempdir'  # Update this path
save_dir = '/data/dhruv_gautam/flickr30k_images/flickr30k_images/tempdir'
os.makedirs(save_dir, exist_ok=True)

# List all JPG files in the source directory
all_files = glob.glob(os.path.join(source_dir, '*.jpg'))
# Select 1000 random files
selected_files = random.sample(all_files, 1000)

for i, file_path in enumerate(selected_files):
    image = Image.open(file_path)
    red_enhanced_image = EnhanceRedTransform()(image)

    # Creating a generic label name since we don't have classes
    label_name = 'image'
    original_image_dir = os.path.join(save_dir, label_name, 'original')
    os.makedirs(original_image_dir, exist_ok=True)
    
    # Use the filename from the original path to avoid overwriting
    filename = os.path.basename(file_path)
    original_image_path = os.path.join(original_image_dir, filename)
    image.save(original_image_path)
    
    red_enhanced_image_dir = os.path.join(save_dir, label_name, 'red_enhanced')
    os.makedirs(red_enhanced_image_dir, exist_ok=True)
    red_enhanced_image_path = os.path.join(red_enhanced_image_dir, filename)
    red_enhanced_image.save(red_enhanced_image_path)

print(f"Images saved to {save_dir}")