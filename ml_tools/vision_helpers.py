import os
import imghdr
from PIL import Image, ImageOps
from typing import Literal
from torchvision import transforms
import torch
from .utilities import _script_info


__all__ = [
    "inspect_images",
    "image_augmentation",
    "ResizeAspectFill",
    "is_image",
    "model_predict"
]


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
        if not is_image(filename):
            non_image.add(filename)
            continue
        # if imghdr.what(filepath) is None:
        #     non_image.add(filename)
        #     continue

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
            filename_no_ext = os.path.splitext(filename)[0]
            new_img.save(f"{dir_name}/{filename_no_ext}_{i}.{output}")
    
    # Print non-image files
    if len(non_image) != 0:
        print(f"Files not processed: {non_image}")


class ResizeAspectFill:
    """
    Custom transformation to make a square image (width/height = 1). 
    
    Implemented by padding with a `pad_color` border an image of size (w, h) when w > h or w < h to match the longest side.
    """
    def __init__(self, pad_color: Literal["black", "white"]="black") -> None:
        self.pad_color = pad_color
        
    def __call__(self, image: Image.Image):
        # Check correct PIL.Image file
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL.Image.Image, got {type(image).__name__}")
        
        w = image.width
        h = image.height
        delta = abs(w - h)
        
        if w > h:
            # padding: left, top, right, bottom
            padding = (0, 0, 0, delta)
        elif h > w: 
            padding = (0, 0, delta, 0)
        else:
            padding = (0, 0)
        
        return ImageOps.expand(image=image, border=padding, fill=self.pad_color)


def is_image(file: str):
    """
    Returns `True` if the file is an image, `False` otherwise.
    
    Args:
       `file`, filename with extension.
    """
    try:
        Image.open(file)
    except IOError:
        return False
    else:
        return True
    
    
def model_predict(model: torch.nn.Module, kind: Literal["regression", "classification"], samples_list: list[torch.Tensor],
                  device: Literal["cpu", "cuda", "mps"]='cpu', view_as: tuple[int,int]=(1,-1), add_batch_dimension: bool=True):
    """
    Returns a list containing lists of predicted values, one for each input sample. 
    
    Each sample must be a tensor and have the same shape and normalization expected by the model. 

    Args:
        `model`: A trained PyTorch model.
        
        `kind`: Regression or Classification task.
    
        `samples_list`: A list of input tensors.
        
        `device`: Device to use, default is CPU.
        
        `view_as`: Reshape each model output, default is (1,-1).
        
        `add_batch_dimension`: Automatically adds the batch dimension to each sample shape.
    """
    # Validate device
    if device == "cuda":
        if not torch.cuda.is_available():
            print("CUDA not available, switching to CPU.")
            device = "cpu"
    elif device == "mps":
        if not torch.backends.mps.is_available():
            print("MPS not available, switching to CPU.")
            device = "cpu"
    
    model.eval()
    results = list()
    with torch.no_grad():
        for data_point in samples_list:
            if add_batch_dimension:
                data_point = data_point.unsqueeze(0).to(device)
            else:
                data_point = data_point.to(device)
                
            output = model(data_point)
            if kind == "classification":
                results.append(output.argmax(dim=1).view(view_as).cpu().tolist())
            else:  #regression
                results.append(output.view(view_as).cpu().tolist())
    
    return results


def info():
    _script_info(__all__)
