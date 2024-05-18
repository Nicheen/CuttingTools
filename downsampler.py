import os
import cv2
import numpy as np

def resize_images(input_folder, output_folder, scale_factor):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the input folder
    files = os.listdir(input_folder)

    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png', ".bmp")):  # Process only image files
            # Read image
            image = cv2.imread(os.path.join(input_folder, file))

            # Resize image
            resized_image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)

            # Write resized image to output folder
            output_path = os.path.join(output_folder, file)
            cv2.imwrite(output_path, resized_image)

input_folder = "G:/Databases/Yellow"
output_base_folder = "G:/Databases/"

# Define scale factors and corresponding output folder names
scales = {
    10: "Yellow_smaller_10x",
    8: "Yellow_smaller_8x",
    6: "Yellow_smaller_6x",
    4: "Yellow_smaller_4x",
    2: "Yellow_smaller_2x",
    1: "Yellow_smaller_1x"
} 

# Iterate over each scale and create the output folder if it doesn't exist
for scale, output_folder_name in scales.items():
    output_folder = os.path.join(output_base_folder, output_folder_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Resize images for the current scale
    resize_images(input_folder, output_folder, 1 / scale)
