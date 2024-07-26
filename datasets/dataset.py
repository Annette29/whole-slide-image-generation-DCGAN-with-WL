import os
import torch
from PIL import Image
import torchvision.transforms as transforms

class Dataset:
    '''
    Dataset class for iteration of training images with data augmentation.
    img_dir: directory to the images
    transforms: augmentation functions
    '''
    def __init__(self, img_dir, transform=None):
        self.images = [
            os.path.join(img_dir, img_path) for img_path in os.listdir(img_dir)
        ]
        print(f'Number of training images: {len(self.images)}')
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        with Image.open(img_path).convert('RGB') as image:
            if self.transform:
                # Apply random horizontal flip
                if torch.rand(1) < 0.5:
                    image = transforms.functional.hflip(image)
                # Apply random cropping
                i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(256, 256))
                image = transforms.functional.crop(image, i, j, h, w)

                image = self.transform(image)

        return image
