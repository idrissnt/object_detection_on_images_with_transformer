import os
import glob
import matplotlib.pyplot as plt
import pydicom
import numpy as np
from PIL import Image
from tqdm import tqdm
from PIL import ImageOps
import pandas as pd

# Threshold for detecting 'white' lungs (adjust based on your data)
white_lung_threshold = 180  # Adjust this value based on experimentation

# Function to check if lungs are white
def are_lungs_white(image_array, threshold=white_lung_threshold):

    # Calculate the average pixel intensity of the image
    avg_intensity = np.mean(image_array)

    # If the average intensity is above the threshold, consider the lungs as white
    return avg_intensity > threshold

# Function to convert DICOM to a PIL image
def dicom_to_pil(dicom_file):
    # Read the DICOM file
    dicom = pydicom.dcmread(dicom_file)

    # Convert DICOM pixel data to a NumPy array
    image_array = dicom.pixel_array.astype(float)

    # Normalize the pixel values to the 0-255 range (8-bit)
    image_minos_min = image_array - np.min(image_array)
    rescaled_image = (image_minos_min / (np.max(image_array) - np.min(image_array))* 255)

    # Convert the NumPy array to a uint8 (8-bit) type
    final_image_array = np.uint8(rescaled_image)

    # Convert the NumPy array to a PIL image
    pil_image = Image.fromarray(final_image_array)

    # Enhance the contrast by scaling pixel values
    enhanced_pil_image =  ImageOps.autocontrast(pil_image, cutoff=5) # Removing 5% of pixel intensity extremes

    return enhanced_pil_image, final_image_array

input_folder = 'data_1/images_archimed/*/*'
list_dicom_files_path = sorted(glob.glob(input_folder)) # type == list

output_folder = 'data_1/new_pil_images'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

with tqdm(total=len(list_dicom_files_path)) as pbar:
    for dicom_file_path in list_dicom_files_path:

        dicom_file_name = dicom_file_path.split('/')[-1]

        # Convert DICOM to a PIL image and get the corresponding NumPy array
        pil_image, image_array = dicom_to_pil(dicom_file_path)

        # Convert the PIL image to grayscale if not already
        pil_image = pil_image.convert('L')

        # Check if lungs are white
        if not are_lungs_white(image_array):
            # Invert the image if lungs are white
            inverted_image = Image.fromarray(255 - np.array(pil_image))
            # print(f"Inverting {dicom_file_name} due to white lungs.")

        else:
            # Keep the image unchanged if lungs are already black
            inverted_image = pil_image
            # print(f"No change for {dicom_file_name}, lungs are already dark.")

        output_image_path = dicom_file_path.split('/')[-2:]
        
        output_folder_image = f'data_1/new_pil_images/{output_image_path[0]}'
        if not os.path.exists(output_folder_image):
            os.makedirs(output_folder_image)

        # Save the processed image in PNG format
        output_image_path = os.path.join(output_folder_image, f'{output_image_path[1]}.png')

        inverted_image.save(output_image_path)
        pbar.update()

print("Processing completed.")

# def load_and_convert_dicom_to_pil_image(dicom_file_path: str) -> Image.Image:

#     """Load a DICOM file and convert it to a PIL image."""

#     # Load a DICOM file
#     dicom_data = pydicom.dcmread(dicom_file_path)
#     image_data = dicom_data.pixel_array

#     # plt.figure()
#     # plt.imshow(image_data, cmap='gray')
#     # plt.title('inital DICOM image')
#     # plt.savefig('wkdir/pic/initial_img.png')
#     # plt.close()

#     # Normalize the pixel values to the 0-255 range (8-bit)
#     lower = np.percentile(image_data, 1)
#     upper = np.percentile(image_data, 98)
#     image_data =  np.clip(image_data, lower, upper)
#     image_data = image_data - np.min(image_data)
#     image_data = (image_data / (np.max(image_data) - np.min(image_data))* 255).astype(np.uint8)

#     # plt.figure()
#     # plt.imshow(image_data, cmap='gray')
#     # plt.title('DICOM convert to PIL image and enhanced')
#     # plt.savefig('wkdir/pic/before_inverse.png')
#     # plt.close()

#     # Convert the numpy array to a PIL image
#     pil_image = Image.fromarray(image_data)

#     #convert to grayscale (ensuring it is grayscale)
#     new_image_gray = pil_image.convert('L')

#     # Invert the grayscale image to give it an X-ray look
#     inverted_image = ImageOps.invert(new_image_gray)

#     # Enhance the contrast by scaling pixel values
#     enhanced_image =  ImageOps.autocontrast(inverted_image, cutoff=5)  # Removing 5% of pixel intensity extremes

#     first_dir = dicom_file_path.split('/')[-2]
#     second_dir = dicom_file_path.split('/')[-1]

#     dir_to_save = 'data_1/not_inverted_image_png'
#     if not os.path.exists(f'{dir_to_save}/{first_dir}'):
#         os.makedirs(f'{dir_to_save}/{first_dir}')

#     plt.figure()
#     plt.imshow(enhanced_image)
#     # plt.imshow(enhanced_image, cmap='gray')
#     plt.title('DICOM convert to PIL image and enhanced')
#     plt.savefig(f'{dir_to_save}/{first_dir}/{second_dir}.png')
#     plt.close()

#     return enhanced_image