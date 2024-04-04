import os
import imghdr
from PIL import Image
from typing import Literal, Union
from torchvision import transforms


# --- Helper Functions ---
def inspect_images(path: str):
    """
    Prints out the types, sizes and channels of image files found in the directory and its subdirectories.
    
    Possible band names (channels):
        * “R”: Red channel
        * “G”: Green channel
        * “B”: Blue channel
        * “A”: Alpha (transparency) channel
        * “L”: Luminance (grayscale) channel
        * “P”: Palette channel
        * “I”: Integer channel
        * “F”: Floating point channel

    Args:
        path (string): path to target directory.
    """
    # Non-image files present?
    red_flag = False
    non_image = set()
    # Image types found
    img_types = set()
    # Image sizes found
    img_sizes = set()
    # Color channels found
    img_channels = set()
    # Number of images
    img_counter = 0
    # Loop through files in the directory and subdirectories
    for root, directories, files in os.walk(path):
        for filename in files:
            filepath = os.path.join(root, filename)
            img_type = imghdr.what(filepath)
            # Not an image file
            if img_type is None:
                red_flag = True
                non_image.add(filename)
                continue
            # Image type
            img_types.add(img_type)
            # Image counter
            img_counter += 1
            # Image size
            img = Image.open(filepath)
            img_sizes.add(img.size)
            # Image color channels 
            channels = img.getbands()
            for code in channels:
                img_channels.add(code)
    
    if red_flag:
        print(f"⚠️ Non-image files found: {non_image}")
    # Print results
    print(f"Image types found: {img_types}\nImage sizes found: {img_sizes}\nImage channels found: {img_channels}\nImages found: {img_counter}")


def image_augmentation(path: str, samples: int=100, size: int=256, mode: Literal["RGB", "L"]="RGB", jitter_ratio: float=0.0, 
                       rotation_deg=270, output: Literal["jpeg", "png", "tiff", "bmp"]="jpeg"):
    """
    Perform image augmentation on a directory containing image files. 
    A new directory "temp_augmented_images" will be created; an error will be raised if it already exists.

    Args:
        path (str): Path to target directory.
        samples (int, optional): Number of images to create per image in the directory. Defaults to 100.
        size (int, optional): Image size to resize to. Defaults to 256.
        mode (str, optional): 'RGB' for 3 channels, 'L' for 1 grayscale channel.
        jitter_ratio (float, optional): Brightness and Contrast factor to use in the ColorJitter transform. Defaults to 0.
        rotation_deg (int, optional): Range for the rotation transformation. Defaults to 270.
        output (str, optional): output image format. Defaults to 'jpeg'.
    """
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize(size=(int(size*1.2),int(size*1.2))),
        transforms.CenterCrop(size=size),
        transforms.ColorJitter(brightness=jitter_ratio, contrast=jitter_ratio), 
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=rotation_deg),
    ])

    # Create container folder
    dir_name = "temp_augmented_images"
    os.makedirs(dir_name, exist_ok=False)
    
    # Keep track of non-image files
    non_image = set()
    
    # Apply transformation to each image in path
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        
        # Is image file?
        if imghdr.what(filepath) is None:
            non_image.add(filename)
            continue

        # current image
        img = Image.open(filepath)
        
        # Convert to RGB or grayscale
        if mode == "RGB":
            img = img.convert("RGB")
        else:
            img = img.convert("L")
        
        # Create and save images
        for i in range(1, samples+1):
            new_img = transform(img)
            new_img.save(f"{dir_name}/{filename}_{i}.{output}")
    
    # Print non-image files
    if len(non_image) != 0:
        print(f"Files not processed: {non_image}")

