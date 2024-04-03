import os
import torchvision
from torchvision.datasets import STL10
from PIL import Image

class EnhanceRedTransform(object):
    def __init__(self, enhanced_red_factor=2.5):
        self.enhanced_red_factor = enhanced_red_factor

    def __call__(self, img):
        r, g, b = img.split()
        enhanced_red = r.point(lambda i: i * self.enhanced_red_factor)
        return Image.merge('RGB', (enhanced_red, g, b))

# Load the STL-10 dataset without applying ToPILImage transform
stl_dataset = STL10(root='./data', split='train', download=True)

# Directory to save the images
save_dir = './stl10_images'
os.makedirs(save_dir, exist_ok=True)

# Process and save the images
for i, (img, _) in enumerate(stl_dataset):
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
    
    if i % 100 == 0:
        print(f"Processed {i+1} images...")

print("Completed processing STL-10 images.")
