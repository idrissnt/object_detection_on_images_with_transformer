
import torch
from transformers import AutoModel
from transformers import AutoImageProcessor

# Initialize the processor and model
repo = "microsoft/rad-dino"
model = AutoModel.from_pretrained(repo)
processor = AutoImageProcessor.from_pretrained(repo)

print(model)












# import torch
# from transformers import AutoModel
# from transformers import AutoImageProcessor

# import pydicom
# from PIL import Image
# import numpy as np

# def load_dicom_image(dicom_file_path: str) -> Image.Image:
#     """Load a DICOM file and convert it to a PIL image."""
#     dicom_data = pydicom.dcmread(dicom_file_path)
#     image_data = dicom_data.pixel_array

#     # Normalize the pixel values to the 0-255 range (8-bit)
#     image_data = image_data - np.min(image_data)
#     image_data = (image_data / np.max(image_data) * 255).astype(np.uint8)

#     # Convert the numpy array to a PIL image
#     pil_image = Image.fromarray(image_data)

#     return pil_image

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
