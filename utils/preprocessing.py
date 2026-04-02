import numpy as np
from PIL import Image

TARGET_SIZE = (224, 224)


def preprocess_image(image_file, preprocess_type="mobilenet_v3"):
    """Load and preprocess an image for model inference.

    Args:
        image_file: File-like object or path to image
        preprocess_type: One of 'mobilenet_v3', 'resnet50', 'vgg16'

    Returns:
        numpy array of shape (1, 224, 224, 3) ready for model.predict()
    """
    try:
        img = Image.open(image_file).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid image file: {e}")
    img = img.resize(TARGET_SIZE)
    img_array = np.array(img, dtype=np.float32)

    if preprocess_type == "mobilenet_v3":
        # Scale to [-1, 1]
        img_array = (img_array / 127.5) - 1.0
    elif preprocess_type == "resnet50":
        # ImageNet mean subtraction (BGR order for caffe mode)
        img_array = img_array[..., ::-1]  # RGB to BGR
        img_array[..., 0] -= 103.939
        img_array[..., 1] -= 116.779
        img_array[..., 2] -= 123.68
    elif preprocess_type == "vgg16":
        # Same as ResNet (caffe mode)
        img_array = img_array[..., ::-1]
        img_array[..., 0] -= 103.939
        img_array[..., 1] -= 116.779
        img_array[..., 2] -= 123.68
    else:
        # Default: scale to [0, 1]
        img_array = img_array / 255.0

    return np.expand_dims(img_array, axis=0)
