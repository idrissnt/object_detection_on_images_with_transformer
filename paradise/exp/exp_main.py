import logging

from PIL import Image
import pandas as pd
from transformers import AutoImageProcessor

from models.vit_rad_dino import CustomDinoModel 
from models.resnet import resnet34
from models.vision_trans import initial_vit
from models.cheXNet import pretrained_model
from utils import RunningAverage

from itertools import chain
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.determinstic = True
torch.multiprocessing.set_sharing_strategy('file_system')

import warnings
warnings.filterwarnings('ignore')


class Exp_Main(object):
    def __init__(self, args):

        self.args = args
        self.device = args.device
        model_dict = {
            'Rad_Dino': CustomDinoModel(),
            'ResNet': resnet34(),
            'initial_vit' : initial_vit,
            'cheXNet' : pretrained_model
        }

        self.model = model_dict[self.args.model].to(self.device)
        self.model_optim = self._select_optimizer()

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _calc_loss_batch(self, input_batch, target_batch, target_csi_regions, model):

        logit_class, logit_csi_scores, logit_mean_csi = model(input_batch)

        loss_csi_score = nn.functional.mse_loss(logit_csi_scores, target_csi_regions)
        loss_class = nn.functional.cross_entropy(logit_class, target_batch)

        loss = loss_class + loss_csi_score

        return loss

    def validation(self, vali_loader):
        print()
        logging.info("Validation...")
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            nan_val = 0
            with tqdm(total=len(vali_loader)) as pbar:
                    
                for samples_batch in vali_loader:

                    input_img = Variable(samples_batch[0]).to(self.device)
                    target_class = Variable(samples_batch[1]).to(self.device)
                    target_csi_regions = Variable(samples_batch[2]).type(torch.FloatTensor).to(self.device)

                    if torch.isnan(input_img).any():
                        nan_val=+1
                        continue

                    loss = self._calc_loss_batch(input_img, target_class, target_csi_regions, self.model)

                    total_loss.append(loss)
                    pbar.update()

            total_loss = torch.mean(torch.tensor(total_loss))
            logging.info("- Eval loss : {:05.3f}".format(total_loss.item()))
            logging.info("- number of files used for validation which contain NaN value : {}".format(nan_val))
            return total_loss

    def train_and_validation(self, batch_train_data, batch_val_data):

        logging.info("Training...")
        loss_avg = RunningAverage()
        self.model.train()

        with tqdm(total=len(batch_train_data)) as pbar:
                
            nan_val = 0
            for i, samples_batch in enumerate(batch_train_data): 

                input_img = Variable(samples_batch[0]).to(self.device)
                target_class = Variable(samples_batch[1]).to(self.device)
                target_csi_regions = Variable(samples_batch[2]).type(torch.FloatTensor).to(self.device)

                # input_chexnet, label, csi_regions, mean_csi, classe = []

                if torch.isnan(input_img).any():
                    nan_val=+1
                    continue
                
                loss = self._calc_loss_batch(input_img, target_class, target_csi_regions, self.model)

                # back propagration : calculating the gradients
                loss.backward(retain_graph=True) 

                # Update model parameters
                self.model_optim.step() 
            
                # Zero gradient
                self.model_optim.zero_grad()

                # progress bar
                loss_avg.update(loss.item()) 
                pbar.set_postfix(loss='{:05.3f}'.format(loss_avg()))
                pbar.update()   

            logging.info("- number of files used for training which contain NaN value : {}".format(nan_val))
            # logging.info("Loss parameters: {}".format(list(loss.parameters())))
            logging.info("Loss train: {:05.3f}".format(loss_avg()))

        train_loss = loss_avg()

        validation_loss = self.validation(batch_val_data)

        print()
        logging.info('Computing accuracy...')
        train_accuracy, class_target_train, class_pred_train = self.calc_accuracy_loader(batch_train_data, self.model, self.device, 'Training', [], [])
        val_accuracy, class_target_val, class_pred_val = self.calc_accuracy_loader(batch_val_data, self.model, self.device, 'Validation', [], [])

        df = pd.DataFrame()
        df['class_target_train'] = list(chain(*class_target_train))
        df['class_pred_train'] = list(chain(*class_pred_train))

        df_val = pd.DataFrame()
        df_val['class_target_val'] = list(chain(*class_target_val))
        df_val['class_pred_val'] = list(chain(*class_pred_val))

        df.to_csv('train_classes.csv')
        df_val.to_csv('val_classes.csv')

        logging.info("Train accuracy: {:.2f}%".format(train_accuracy*100))
        logging.info("Val accuracy: {:.2f}%".format(val_accuracy*100))

        return train_loss, validation_loss, train_accuracy, val_accuracy
        
    def calc_accuracy_loader(self, data_loader, model, device, accuracy_name, class_target:list, class_pred:list):

        model.to(device)

        model.eval()
        correct_predictions, num_examples = 0, 0

        num_batches = len(data_loader)

        with tqdm(total=len(data_loader)) as pbar:
            for i, samples_batch  in enumerate(data_loader):
                input_batch, target_batch = samples_batch[0], samples_batch[1]
                if i < num_batches:
                    input_batch = input_batch.to(device)
                    target_batch = target_batch.to(device)

                    with torch.no_grad():
                        logit_class, logit_csi_scores, logit_mean_csi = model(input_batch)
                    # proba = torch.softmax(logits, dim=-1)
                    predicted_labels = torch.argmax(logit_class, dim=-1)

                    class_target.append(target_batch.cpu().numpy().tolist())
                    class_pred.append(predicted_labels.cpu().numpy().tolist())

                    num_examples += predicted_labels.shape[0]
                    correct_predictions += (
                        (predicted_labels == target_batch).sum().item()
                    )

                else:
                    break

                pbar.update()

        accuracy = correct_predictions / num_examples
        # print(f"{accuracy_name} accuracy: {accuracy*100:.2f}%")

        return accuracy, class_target, class_pred
    
    def calc_loss_loader(self, data_loader, model, device):

        model.to(device)
        total_loss = 0.

        if len(data_loader) == 0:
            return float("nan")
        
        num_batches = len(data_loader)

        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = self._calc_loss_batch(
                    input_batch, target_batch, model, device
                )
                total_loss += loss.item()
            else:
                break
        return total_loss / num_batches
    
    def classify_review(self, model, dicom_file_path, data_dirr_classes):

        model.eval()

        # dicom_file_path = 'data_1/new_pil_images/2020-128 01-0001/1.2.840.113619.2.203.4.2147483647.1420095596.215360.png'
        # data_dirr_classes = 'data_1/labels/paradise_csi_w_classes.csv'

        pil_images = Image.open(dicom_file_path)

        # Initialize the processor
        repo = "microsoft/rad-dino"
        # repo = "google/vit-base-patch16-224-in21k"
        processor = AutoImageProcessor.from_pretrained(repo)

        # Preprocess the DICOM image 
        """The processor takes a PIL image, performs resizing, center-cropping, and
        intensity normalization using stats from MIMIC-CXR, and returns a
        dictionary with a PyTorch tensor ready for the encoder"""
        inputs_dic = processor(images=pil_images, return_tensors="pt")
        input_tensor = inputs_dic['pixel_values'].squeeze()

        with torch.no_grad():
            logits = model(input_tensor)[:, -1, :]
        predicted_label = torch.argmax(logits, dim=-1).item()

        def get_imag_class():

            image_class_df = pd.read_csv(data_dirr_classes) 

            number_img = dicom_file_path.split('/')[-2].split('-')[-1]
            number_df = image_class_df[image_class_df.number == int(number_img)]

            classe_label = list(number_df.classes_label)[0]
            class_ = list(number_df.classes)[0]

            return classe_label, class_, number_img
        
        x = get_imag_class()

        return x, "AHF_risk = 4.4%" if predicted_label == 0 else "AHF_risk = 25.8%" if predicted_label == 1 else 'AHF_risk = 60.3%'
    
    def compute_AUCs(gt, pred):
        """Computes Area Under the Curve (AUC) from prediction scores.

        Args:
            gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
            true binary labels.
            pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
            can either be probability estimates of the positive class,
            confidence values, or binary decisions.

        Returns:
            List of AUROCs of all classes.
        """

        n_classes = 3
        AUROCs = []
        gt_np = gt.cpu().numpy()
        pred_np = pred.cpu().numpy()
        for i in range(n_classes):
            AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))

        return AUROCs

    def get_performance_per_class(self, testloader, model, device):

        classes = ['AHF_risk = 4.4%', 'AHF_risk = 25.8%', 'AHF_risk = 60.3%']

        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        # again no gradients needed
        with torch.no_grad():
            with tqdm(total=len(testloader*16)) as pbar:
                for samples_batch in testloader:

                    input_img = Variable(samples_batch[0]).to(device)
                    label = Variable(samples_batch[1]).to(device)
                    # target_class = samples_batch[3][0]
                    
                    logits = model(input_img)

                    # collect the correct predictions for each class
                    for label, predicted_label in zip(label, logits):

                        predicted_label = torch.argmax(predicted_label, dim=-1).item()
                        if  predicted_label == label.item():
                            correct_pred[classes[label]] += 1
                        total_pred[classes[label]] += 1
                        pbar.update()

                print(correct_pred)
                print(total_pred)


        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')        

