import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
import pandas as pd
from transformers import AutoImageProcessor


class XRayDataset(Dataset):
    def __init__(self):

        data_dirr = 'data_1/images_archimed/*/*' # for the directory 

        self.dicom_files_path = glob.glob(data_dirr) # type == list
        self.image_scores = pd.read_csv('data_1/labels/paradise_csi_drop_nan.csv')

    def __len__(self):
        return len(self.dicom_files_path)

    def __getitem__(self, idx):

        print(len(self.dicom_files_path))

        dicom_file_path = self.dicom_files_path[idx]

        #get scores
        dicom_file_path, scores = get_imag_scores(dicom_file_path, self.image_scores)

        # # Load your DICOM image instead of downloading a sample image
        image = load_dicom_image(dicom_file_path)

        # Check the size of the image
        # print(image.size)  # (width, height)

        # Initialize the processor
        repo = "microsoft/rad-dino"
        processor = AutoImageProcessor.from_pretrained(repo)

        # Preprocess the DICOM image
        inputs = processor(images=image, return_tensors="pt")
        outputs = torch.tensor(scores, dtype=torch.float)

        return inputs, outputs

def get_imag_scores(img_path, df_scores):

    number_img = img_path.split('/')[-2].split('-')[-1]
    number_df = df_scores[df_scores.number == int(number_img)]

    right_sup ,left_sup = list(number_df.right_sup)[0] , list(number_df.left_sup)[0] 
    right_mid ,left_mid = list(number_df.right_mid)[0] ,list(number_df.left_mid)[0]

    print(img_path, [right_sup ,left_sup ,right_mid ,left_mid])

    return img_path, [right_sup ,left_sup ,right_mid ,left_mid]

def load_dicom_image(dicom_file_path: str) -> Image.Image:

    """Load a DICOM file and convert it to a PIL image."""
    dicom_data = pydicom.dcmread(dicom_file_path)
    image_data = dicom_data.pixel_array

    # Normalize the pixel values to the 0-255 range (8-bit)
    image_data = image_data - np.min(image_data)
    image_data = (image_data / np.max(image_data) * 255).astype(np.uint8)

    # Convert the numpy array to a PIL image
    pil_image = Image.fromarray(image_data)

    return pil_image


# Example: Dataloader
dataset = XRayDataset()
x , y = dataset[100]
print(x, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)















# # Load your DICOM image instead of downloading a sample image
# dicom_file_path = "data_1/data_archimed/2020-128 01-0002/1.3.51.0.7.11324537245.60188.18408.42903.58162.20795.51958"
# image = load_dicom_image(dicom_file_path)

# # Check the size of the image
# print(image.size)  # (width, height)

# # Initialize the processor and model
# repo = "microsoft/rad-dino"
# model = AutoModel.from_pretrained(repo)
# processor = AutoImageProcessor.from_pretrained(repo)

# # # Preprocess the DICOM image
# # inputs = processor(images=image, return_tensors="pt")
# # # print(type(inputs))

# # # Encode the image using the model
# # with torch.inference_mode():
# #     outputs = model(**inputs)

# # print(outputs.last_hidden_state.shape)
# # print(outputs)

# # # Get the CLS embeddings
# # cls_embeddings = outputs.pooler_output
# # # print(cls_embeddings.shape)  # (batch_size, num_channels)


# # # for name, module in model.named_children():
# # #     print(f"Layer Name: {name}")
# # #     print(module)
# # #     print("=" * 50)
