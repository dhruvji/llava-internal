import os
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection

class EnhanceRedTransform(object):
    def __init__(self, enhanced_red_factor=1.5):
        self.enhanced_red_factor = enhanced_red_factor

    def __call__(self, img):
        r, g, b = img.split()
        enhanced_red = r.point(lambda i: i * self.enhanced_red_factor)
        return Image.merge('RGB', (enhanced_red, g, b))

# Paths for COCO dataset (adjust as necessary)
coco_root = './data/coco'
annFile = f'{coco_root}/annotations/instances_train2017.json'

# Load a subset of the COCO dataset
coco_dataset = CocoDetection(root=f'{coco_root}/train2017',
                             annFile=annFile,
                             transform=transforms.ToPILImage())

subset_size = 100  # Define the size of your subset here

# Directory to save the images
save_dir = './coco_images_subset'
os.makedirs(save_dir, exist_ok=True)

# Process only the subset
for i, (img, _) in enumerate(coco_dataset):
    if i >= subset_size:  # Stop after processing subset_size images
        break
    
    # Original image save path
    original_image_dir = os.path.join(save_dir, 'original')
    os.makedirs(original_image_dir, exist_ok=True)
    img.save(os.path.join(original_image_dir, f'{i}.jpg'))
    
    # Red-enhanced image
    red_enhanced_image = EnhanceRedTransform()(img)
    
    # Red-enhanced image save path
    red_enhanced_image_dir = os.path.join(save_dir, 'red_enhanced')
    os.makedirs(red_enhanced_image_dir, exist_ok=True)
    red_enhanced_image.save(os.path.join(red_enhanced_image_dir, f'{i}.jpg'))
    
    if i % 10 == 0:
        print(f"Processed {i+1} images...")

print(f"Completed processing {subset_size} COCO images.")
