# data/image_reader.py

import cv2


def read_image(image_path, image_size=(32, 32)):
    """
    Reads an RGB image, resizes it, normalizes it,
    and returns channel-first data.

    Returns:
        image_data: list with shape [3][H][W]
    """

    # Read image in color (BGR)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    # Resize image
    img = cv2.resize(img, image_size)

    height, width, channels = img.shape  # channels = 3

    # Convert to channel-first format [C][H][W]
    image_data = []

    for c in range(channels):
        channel = []
        for i in range(height):
            row = []
            for j in range(width):
                row.append(img[i, j, c] / 255.0)
            channel.append(row)
        image_data.append(channel)

    return image_data
