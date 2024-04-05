import os
from PIL import Image
import random
import glob

class EnhanceRedTransform(object):
    def __init__(self, enhanced_red_factor=4.5):
        self.enhanced_red_factor = enhanced_red_factor

    def __call__(self, img):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        r, g, b = img.split()
        enhanced_red = r.point(lambda i: i * self.enhanced_red_factor)
        return Image.merge('RGB', (enhanced_red, g, b))

source_dir = '/data/dhruv_gautam/coco/val2017'  # Update this path
save_dir = '/data/dhruv_gautam/coco/val2017'
os.makedirs(save_dir, exist_ok=True)

all_files = glob.glob(os.path.join(source_dir, '*.jpg'))

for i, file_path in enumerate(all_files):
    image = Image.open(file_path)
    red_enhanced_image = EnhanceRedTransform()(image)

    label_name = 'image'
    original_image_dir = os.path.join(save_dir, label_name, 'original')
    os.makedirs(original_image_dir, exist_ok=True)
    
    filename = os.path.basename(file_path)
    original_image_path = os.path.join(original_image_dir, filename)
    image.save(original_image_path)
    
    red_enhanced_image_dir = os.path.join(save_dir, label_name, 'red_enhanced')
    os.makedirs(red_enhanced_image_dir, exist_ok=True)
    red_enhanced_image_path = os.path.join(red_enhanced_image_dir, filename)
    red_enhanced_image.save(red_enhanced_image_path)

print(f"Images saved to {save_dir}")