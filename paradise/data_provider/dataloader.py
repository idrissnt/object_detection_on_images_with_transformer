import logging
from joblib import Memory
import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
from PIL import Image
import glob
import pandas as pd
from transformers import AutoImageProcessor

class XRayDataset(Dataset):
    def __init__(self):

        data_dirr_mage = 'data_1/images_archimed/*/*'
        data_dirr_scores = 'data_1/labels/paradise_csi_drop_nan.csv'

        dicom_files_path = sorted(glob.glob(data_dirr_mage)) # type == list

        self.image_scores = pd.read_csv(data_dirr_scores)
        self.dicom_files_path = sorted(list(set([fname for fname in dicom_files_path if 
                                 int(fname.split('/')[-2].split(' ')[-1].split('-')[-1]) in list(self.image_scores.number)])))
        
        self.patient_ids = sorted(list(set([fname.split('/')[-2] for fname in self.dicom_files_path])))

    def __len__(self):
        return len(self.dicom_files_path)

    def __getitem__(self, idx):

        dicom_file_path = self.dicom_files_path[idx]
        dicom_file_path, scores = get_imag_scores(dicom_file_path, self.image_scores)
        image = load_dicom_image(dicom_file_path)

        # Initialize the processor
        repo = "microsoft/rad-dino"
        processor = AutoImageProcessor.from_pretrained(repo)

        # Preprocess the DICOM image
        inputs = processor(images=image, return_tensors="pt")
        outputs = torch.tensor(scores, dtype=torch.float)

        return inputs['pixel_values'].squeeze(0), outputs

def get_imag_scores(img_path, df_scores):

    number_img = img_path.split('/')[-2].split('-')[-1]
    number_df = df_scores[df_scores.number == int(number_img)]

    # print(int(number_img)) 
    # print(df_scores[df_scores.number == int(number_img)])

    right_sup ,left_sup = list(number_df.right_sup)[0] , list(number_df.left_sup)[0] 
    right_mid ,left_mid = list(number_df.right_mid)[0] ,list(number_df.left_mid)[0]

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

# memory = Memory(location='cache_directory', verbose=0) 
# @memory.cache

def get_data(batch_size): 

    data_set = XRayDataset()
    
    patient_ids = data_set.patient_ids
    train_ids, val_ids, test_ids  = random_split(patient_ids, (781, 200, 200))

    #make sure that a each patient id remain a same subsampler
    dicom_files_path_idx_train = [idx for idx, fname in enumerate(data_set.dicom_files_path) if fname.split('/')[-2] in train_ids]
    dicom_files_path_idx_val = [idx for idx, fname in enumerate(data_set.dicom_files_path) if fname.split('/')[-2] in val_ids]
    dicom_files_path_idx_test = [idx for idx, fname in enumerate(data_set.dicom_files_path) if fname.split('/')[-2] in test_ids]

    # Sample elements randomly from a given list of ids, no replacement.   
    train_subsampler = SubsetRandomSampler(dicom_files_path_idx_train)
    val_subsampler = SubsetRandomSampler(dicom_files_path_idx_val)
    test_subsampler = SubsetRandomSampler(dicom_files_path_idx_test)

    logging.info(f'loading X ray images : {len(train_subsampler)} for training, {len(val_subsampler)} for validation and {len(test_subsampler)} for testing...')
    batch_train_data = DataLoader(
            data_set,
            batch_size = batch_size, 
            sampler=train_subsampler)

    batch_val_data = DataLoader(
            data_set,
            batch_size = batch_size, 
            sampler=val_subsampler)
    
    batch_test_data = DataLoader(
            data_set,
            batch_size = batch_size,
            sampler=test_subsampler)

    logging.info(f'done...')

    return batch_train_data, batch_val_data, batch_test_data

# # Example: Dataloader
# dataset = XRayDataset()
# x , y = dataset[100]
# print(x, y)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
