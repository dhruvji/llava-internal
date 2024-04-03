import os
import torchvision
from torchvision.datasets import CIFAR10
from PIL import Image

class EnhanceRedTransform(object):
    def __init__(self, enhanced_red_factor=1.5):
        self.enhanced_red_factor = enhanced_red_factor

    def __call__(self, img):
        r, g, b = img.split()
        enhanced_red = r.point(lambda i: i * self.enhanced_red_factor)
        return Image.merge('RGB', (enhanced_red, g, b))

dataset = CIFAR10(root='./data', train=True, download=True)
save_dir = './cifar10_images'
os.makedirs(save_dir, exist_ok=True)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

for i, (image, label) in enumerate(dataset):
    red_enhanced_image = EnhanceRedTransform()(image)
    label_name = class_names[label]
    original_image_dir = os.path.join(save_dir, label_name, 'original')
    os.makedirs(original_image_dir, exist_ok=True)
    original_image_path = os.path.join(original_image_dir, f"{i}.jpg")
    image.save(original_image_path)
    red_enhanced_image_dir = os.path.join(save_dir, label_name, 'red_enhanced')
    os.makedirs(red_enhanced_image_dir, exist_ok=True)
    red_enhanced_image_path = os.path.join(red_enhanced_image_dir, f"{i}.jpg")
    red_enhanced_image.save(red_enhanced_image_path)

print(f"Images saved to {save_dir}")

