import os
from PIL import Image
import glob

class GreyScaleTransform(object):
    def __call__(self, img):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img.convert('L')  # Convert image to greyscale

source_dir = '/data/dhruv_gautam/coco/val2017'  # Update this path
save_dir = '/data/dhruv_gautam/coco/val2017'
os.makedirs(save_dir, exist_ok=True)

all_files = glob.glob(os.path.join(source_dir, '*.jpg'))

for i, file_path in enumerate(all_files):
    image = Image.open(file_path)
    greyscale_image = GreyScaleTransform()(image)

    label_name = 'image_grey'
    original_image_dir = os.path.join(save_dir, label_name, 'original')
    os.makedirs(original_image_dir, exist_ok=True)
    
    filename = os.path.basename(file_path)
    original_image_path = os.path.join(original_image_dir, filename)
    image.save(original_image_path)
    
    greyscale_image_dir = os.path.join(save_dir, label_name, 'greyscale')
    os.makedirs(greyscale_image_dir, exist_ok=True)
    greyscale_image_path = os.path.join(greyscale_image_dir, filename)
    greyscale_image.save(greyscale_image_path)

print(f"Images saved to {save_dir}")
