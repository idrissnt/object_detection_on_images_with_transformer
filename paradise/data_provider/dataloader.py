import glob
import logging

import torch
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor, transforms


# import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from transformers import AutoImageProcessor
from sklearn.utils.class_weight import compute_class_weight

from tqdm import tqdm
from joblib import Memory

class XRayDataset(Dataset):
    def __init__(self):

        extrat_files = ["0528", "0725", "0760", "0763", "0875", "0933", "0994", "1110", "1111", "1283", "1294", "1329",
                "1349", "1353", "1365", "1424", "1437", "1520", "1708", "1730", "1739", "1741", "1743",
                "1774", "1776", "1821", "1832", "1833", "1834", "1842", "1877", "1891", "1892", "1899",
                "1904", "1911", "1916", "1924", "1930", "1942", "1943", "1959", "1977", "1979", "2017",
                "2022", "2087", "2091", "2157", "2158", "2176", "2185", "2190", "2192", "2196", "2233",
                "2235", "2242", "2255", "2267", "2281"]

        data_dirr_mage = 'data_1/new_pil_images/*/*'
        data_dirr_scores = 'data_1/labels/paradise_csi_w_classes_w_non_nan.csv'
        image_scores_df = pd.read_csv(data_dirr_scores)

        dicom_files_path = sorted(glob.glob(data_dirr_mage)) # type == list
        new_dicom_files_path = [val for val in dicom_files_path if str(val.split('/')[-2].split('-')[-1]) not in extrat_files]
     
        self.image_scores_df = image_scores_df[~image_scores_df.number.isin([int(val) for val in extrat_files])]
        self.dicom_files_path = sorted(list(set([fname for fname in new_dicom_files_path if 
                                 int(fname.split('/')[-2].split(' ')[-1].split('-')[-1]) in list(self.image_scores_df.number)])))
        self.patient_ids = sorted(list(set([fname.split('/')[-2] for fname in self.dicom_files_path])))
        
    def __len__(self):
        return len(self.dicom_files_path)

    def __getitem__(self, idx):

        dicom_file_path = self.dicom_files_path[idx]

        (
            dicom_file_path, classe_label, 
            classe, csi_regions, mean_csi, 
            number_fname, id_number_fname
        ) = get_imag_scores(dicom_file_path, self.image_scores_df)

        pil_image = Image.open(dicom_file_path).convert('RGB')

        # inputs = img_process_google(pil_image) # cheXNet_3
        inputs = img_process_microsoft(pil_image) # cheXNet_2
        # inputs = img_process_chexnet(pil_image)[-1] # cheXNet_1
        
        label_classification = torch.tensor(classe_label, dtype=torch.int64)
        csi_regions = torch.tensor(csi_regions, dtype=torch.float64)
        mean_csi = torch.tensor(mean_csi, dtype=torch.float64)

        return inputs, label_classification, csi_regions, mean_csi, classe, number_fname , id_number_fname

def img_process_microsoft(pil_image):
    # Initialize the processor
    repo = "microsoft/rad-dino"
    processor = AutoImageProcessor.from_pretrained(repo)

    # Preprocess the DICOM image 
    # """The processor takes a PIL image, performs resizing, center-cropping, and
    # intensity normalization using stats from MIMIC-CXR, and returns a
    # dictionary with a PyTorch tensor ready for the encoder"""
    inputs_dic = processor(images=pil_image, return_tensors="pt")
    inputs = inputs_dic['pixel_values'].squeeze()
    
    return inputs

def img_process_chexnet(pil_image):

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    transform=transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda
            (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda
            (lambda crops: torch.stack([normalize(crop) for crop in crops]))
            ])
    
    input_chexnet = transform(pil_image)

    return input_chexnet

def img_process_google(pil_image):

    checkpoint = "google/vit-base-patch16-224-in21k"
    image_processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=True)

    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    size = ( image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"]))
    
    _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])
    pil_image = _transforms(pil_image)

    return pil_image
    
def get_imag_scores(img_path, df_scores):

    number_img = img_path.split('/')[-2].split('-')[-1]
    number_df = df_scores[df_scores.number == int(number_img)]

    csi_regions = [list(number_df[val])[0] for val in ['right_sup','left_sup','right_mid','left_mid','right_inf','left_inf']]
    mean_csi = list(number_df.csi)[0]

    classe_label = list(number_df.classes_label)[0]
    classe = list(number_df.classes)[0]

    number_fname , id_number_fname  = list(number_df.number)[0], list(number_df.id_number)[0]

    return img_path, classe_label, classe, csi_regions, mean_csi, number_fname , id_number_fname

memory = Memory(location='cache_directory_', verbose=0) 
# memory = Memory(location='cache_directory_test', verbose=0) 
@memory.cache

def get_data(batch_size): 

    data_set = XRayDataset()
    
    patient_ids = data_set.patient_ids
    train_ids, val_ids, test_ids  = random_split(patient_ids, (880, 183, 183)) # 1246 # 880

    # make sure that each patient remains in the same subsampler
    dicom_files_path_idx_train = [idx for idx, fname in enumerate(data_set.dicom_files_path) if fname.split('/')[-2] in train_ids]
    dicom_files_path_idx_val = [idx for idx, fname in enumerate(data_set.dicom_files_path) if fname.split('/')[-2] in val_ids]
    dicom_files_path_idx_test = [idx for idx, fname in enumerate(data_set.dicom_files_path) if fname.split('/')[-2] in test_ids]

    # Sample elements randomly from a given list of ids, no replacement.   
    train_subsampler = SubsetRandomSampler(dicom_files_path_idx_train)
    val_subsampler = SubsetRandomSampler(dicom_files_path_idx_val)
    test_subsampler = SubsetRandomSampler(dicom_files_path_idx_test)

    logging.info(f'loading chest X-ray images...')
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
    logging.info(f'Loading {len(train_subsampler)} for training...')
    batch_train_data = [samples_batch for _, samples_batch in enumerate(tqdm(batch_train_data))]
    print()
    logging.info(f'Loading {len(val_subsampler)} for validation...')
    batch_val_data = [samples_batch for _, samples_batch in enumerate(tqdm(batch_val_data))]
    print()
    logging.info(f'Loading {len(test_subsampler)} for testing...')
    batch_test_data = [samples_batch for _, samples_batch in enumerate(tqdm(batch_test_data))]

    print()
    logging.info(f'done...')

    return batch_train_data, batch_val_data, batch_test_data

# dataset = XRayDataset()
# # print(dataset.image_scores_df)
# inputs, label_classification, csi_regions, mean_csi, classe, number_fname , id_number_fname = dataset[5]

# print(inputs.shape, label_classification, csi_regions, mean_csi, classe)
# # print(x.shape, y,  fname)
# # x = [dataset[i] for i in tqdm(range((len(dataset.dicom_files_path))))]

# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
