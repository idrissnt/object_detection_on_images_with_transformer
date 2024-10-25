import os
import glob
import logging

import torch
import pydicom
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageOps
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler

from tqdm import tqdm
from joblib import Memory

class XRayDataset(Dataset):
    def __init__(self, all_data=False):

        extrat_files = ["0528", "0725", "0760", "0763", "0875", "0933", "0994", "1110", "1111", "1283", "1294", "1329",
                "1349", "1353", "1365", "1424", "1437", "1520", "1708", "1730", "1739", "1741", "1743",
                "1774", "1776", "1821", "1832", "1833", "1834", "1842", "1877", "1891", "1892", "1899",
                "1904", "1911", "1916", "1924", "1930", "1942", "1943", "1959", "1977", "1979", "2017",
                "2022", "2087", "2091", "2157", "2158", "2176", "2185", "2190", "2192", "2196", "2233",
                "2235", "2242", "2255", "2267", "2281"]

        data_dirr_mage = 'data_1/new_pil_images/*/*'
        data_dirr_scores = 'data_1/labels/paradise_csi_drop_nan.csv'
        image_scores = pd.read_csv(data_dirr_scores)

        dicom_files_path = sorted(glob.glob(data_dirr_mage)) # type == list
        new_dicom_files_path = [val for val in dicom_files_path if str(val.split('/')[-2].split('-')[-1]) not in extrat_files]
     
        self.image_scores = image_scores[~image_scores.number.isin([int(val) for val in extrat_files])]
        self.dicom_files_path = sorted(list(set([fname for fname in new_dicom_files_path if 
                                 int(fname.split('/')[-2].split(' ')[-1].split('-')[-1]) in list(self.image_scores.number)])))
        self.patient_ids = sorted(list(set([fname.split('/')[-2] for fname in self.dicom_files_path])))

        # Step 1: Calculate class weights using sklearn
        if all_data:
            all_scores = [list(self.image_scores.right_sup), list(self.image_scores.left_sup), list(self.image_scores.right_mid), 
                        list(self.image_scores.left_mid), list(self.image_scores.right_inf), list(self.image_scores.left_inf)]
        else:
            all_scores = [list(self.image_scores.right_sup), list(self.image_scores.left_sup), list(self.image_scores.right_mid), 
                        list(self.image_scores.left_mid)]
            
        self.all_class_weights = [torch.tensor(compute_class_weight(class_weight='balanced', classes=np.array([0, 1, 2, 3]), y=scores), dtype=torch.float)
                            for scores in all_scores]
        
    def __len__(self):
        return len(self.dicom_files_path)

    def __getitem__(self, idx):

        dicom_file_path = self.dicom_files_path[idx]
        dicom_file_path, scores = get_imag_scores(dicom_file_path, self.image_scores)

        pil_images = Image.open(dicom_file_path)

        # Initialize the processor
        repo = "microsoft/rad-dino"
        processor = AutoImageProcessor.from_pretrained(repo)

        # Preprocess the DICOM image 
        """The processor takes a PIL image, performs resizing, center-cropping, and
        intensity normalization using stats from MIMIC-CXR, and returns a
        dictionary with a PyTorch tensor ready for the encoder"""
        inputs_dic = processor(images=pil_images, return_tensors="pt")
        inputs = inputs_dic['pixel_values'].squeeze()
        outputs = torch.tensor(scores, dtype=torch.float)

        # plt.figure
        # plt.imshow(inputs[0], cmap='gray')
        # plt.title('image preprocess for the model input (from rad-dino)')
        # plt.savefig('wkdir/pic/image_process_.png')
        # plt.close()

        return inputs, outputs, dicom_file_path

def get_imag_scores(img_path, df_scores):

    number_img = img_path.split('/')[-2].split('-')[-1]
    number_df = df_scores[df_scores.number == int(number_img)]

    right_sup ,left_sup = list(number_df.right_sup)[0] , list(number_df.left_sup)[0] 
    right_mid ,left_mid = list(number_df.right_mid)[0] ,list(number_df.left_mid)[0]
    # right_inf ,left_inf = list(number_df.right_inf)[0] ,list(number_df.left_inf)[0]

    # return img_path, [right_sup ,left_sup ,right_mid ,left_mid, right_inf ,left_inf]
    return img_path, [right_sup ,left_sup ,right_mid ,left_mid]


memory = Memory(location='cache_directory', verbose=0) 
@memory.cache

def get_data(batch_size): 

    data_set = XRayDataset()
    
    patient_ids = data_set.patient_ids
    # train_ids, val_ids, test_ids  = random_split(patient_ids, (387, 100, 100))
    train_ids, val_ids, test_ids  = random_split(patient_ids, (725, 200, 200))

    # make sure that each patient remains in the same subsampler
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
    print()
    logging.info(f'for training...')
    batch_train_data = [samples_batch for _, samples_batch in enumerate(tqdm(batch_train_data))]
    print()
    logging.info(f'for validation...')
    batch_val_data = [samples_batch for _, samples_batch in enumerate(tqdm(batch_val_data))]
    print()
    logging.info(f'for testing...')
    batch_test_data = [samples_batch for _, samples_batch in enumerate(tqdm(batch_test_data))]

    print()
    logging.info(f'done...')

    return batch_train_data, batch_val_data, batch_test_data, data_set.all_class_weights

# dataset = XRayDataset()
# print(dataset.image_scores)
# x, y, fname = dataset[5]
# print(x.shape, y.shape, fname)
# x = [dataset[i] for i in tqdm(range((len(dataset.dicom_files_path))))]

# # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
