import logging

import numpy as np

# import utils
# from utils.metrics import metric
from models.vit_rad_dino import CustomDinoModel, model 
from models.resnet import resnet34
from models.vision_trans import initial_vit
from utils import RunningAverage

import torch

import torch.nn as nn
from tqdm import tqdm
from torch import optim
from torch.autograd import Variable

from torch.nn import functional as F

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.determinstic = True
torch.multiprocessing.set_sharing_strategy('file_system')

import os
import warnings

warnings.filterwarnings('ignore')


class Exp_Main(object):
    def __init__(self, args):

        self.args = args
        self.device = args.device
        model_dict = {
            'Rad_Dino': CustomDinoModel,
            'ResNet': resnet34(),
            'initial_vit' : initial_vit
        }

        if self.args.model == 'ResNet':
            self.model = model_dict[self.args.model].to(self.device)
        elif self.args.model == 'Rad_Dino': 
            self.model = model_dict[self.args.model](model).to(self.device)
        elif self.args.model == 'initial_vit':
            self.model = model_dict[self.args.model].to(self.device)
        
        self.model_optim = self._select_optimizer()

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, all_class_weights):
        # criterion = nn.MSELoss()

        # w_right_sup, w_left_sup =  all_class_weights[0].to(self.device), all_class_weights[1].to(self.device)
        # w_right_mid, w_left_mid= all_class_weights[2].to(self.device), all_class_weights[3].to(self.device)
        # w_right_inf, w_left_inf = all_class_weights[4].to(self.device), all_class_weights[5].to(self.device)
        
        # criterion_right_sup = nn.CrossEntropyLoss(weight=w_right_sup)
        # criterion_left_sup  = nn.CrossEntropyLoss(weight=w_left_sup)
        # criterion_right_mid = nn.CrossEntropyLoss(weight=w_right_mid)
        # criterion_left_mid  = nn.CrossEntropyLoss(weight=w_left_mid)
        # criterion_right_inf = nn.CrossEntropyLoss(weight=w_right_inf)
        # criterion_left_inf  = nn.CrossEntropyLoss(weight=w_left_inf)

        criterion_right_sup = nn.CrossEntropyLoss()
        criterion_left_sup  = nn.CrossEntropyLoss()
        criterion_right_mid = nn.CrossEntropyLoss()
        criterion_left_mid  = nn.CrossEntropyLoss()
        criterion_right_inf = nn.CrossEntropyLoss()
        criterion_left_inf  = nn.CrossEntropyLoss()

        return [criterion_right_sup, criterion_left_sup, criterion_right_mid, criterion_left_mid, criterion_right_inf, criterion_left_inf]

    def run_model(self, model, x):
        outputs = model(x)
        return outputs

    # def validation(self, model , vali_loader, all_loss_fn):
    #     print()
    #     logging.info("Validation...")
    #     total_loss = []
    #     model.eval()
    #     with torch.no_grad():
    #         nan_val = 0
    #         with tqdm(total=len(vali_loader)) as pbar:
                    
    #             for samples_batch in vali_loader:

    #                 input_img = Variable(samples_batch[0]).to(self.device)
    #                 target_scores = samples_batch[1].type(torch.LongTensor)
    #                 target_scores = Variable(target_scores).to(self.device)
    #                 # target_scores = Variable(samples_batch[1]).to(self.device)

    #                 if torch.isnan(input_img).any():
    #                     nan_val=+1
    #                     continue

    #                 pred_1, pred_2, pred_3, pred_4= self.model(input_img)

    #                 all_pred = [pred_1, pred_2, pred_3, pred_4]
    #                 loss = sum([all_loss_fn[i](all_pred[i], target_scores[:,i]) for i in range(4)])

    #                 all_pred = [torch.argmax(F.softmax(val, dim=1), dim=1) for val in all_pred]
    #                 print(all_pred[0])
    #                 print(target_scores.permute(1,0)[0])

    #                 # pred_1, pred_2, pred_3, pred_4, pred_5, pred_6 = self.model(input_img)
    #                 # all_pred = [pred_1, pred_2, pred_3, pred_4, pred_5, pred_6]
    #                 loss = sum([all_loss_fn[i](all_pred[i], target_scores[:,i]) for i in range(4)])

    #                 total_loss.append(loss)
    #                 pbar.update()

    #         total_loss = torch.mean(torch.tensor(total_loss))
    #         logging.info("- Eval loss : {:05.3f}".format(total_loss.item()))
    #         logging.info("- number of files used for validation which contain NaN value : {}".format(nan_val))
    #         return total_loss
        

    def validation(self, model , vali_loader, all_loss_fn):
        print()
        logging.info("Validation...")
        total_loss = []
        model.eval()
        with torch.no_grad():
            nan_val = 0
            with tqdm(total=len(vali_loader)) as pbar:
                
                all_right_sup_pred , all_left_sup_pred , all_right_mid_pred , all_left_mid_pred = [],[], [], []
                all_right_sup_target , all_left_sup_target , all_right_mid_target , all_left_mid_target = [],[], [], []

                for samples_batch in vali_loader:

                    input_img = Variable(samples_batch[0]).to(self.device)
                    target_scores = samples_batch[1].type(torch.LongTensor)
                    target_scores = Variable(target_scores).to(self.device).permute(1,0)
                    # target_scores = Variable(samples_batch[1]).to(self.device)

                    if torch.isnan(input_img).any():
                        nan_val=+1
                        continue

                    pred_1, pred_2, pred_3, pred_4= self.model(input_img)
                    all_pred = [pred_1, pred_2, pred_3, pred_4]

                    all_pred = [torch.argmax(F.softmax(val, dim=1), dim=1) for val in all_pred]

                    # right_sup_target = target_scores[0]
                    # left_sup_target = target_scores[1]
                    # right_mid_target = target_scores[2]
                    # left_mid_target = target_scores[3]

                    # right_sup_pred = all_pred[0]
                    # left_sup_pred = all_pred[1]
                    # right_mid_pred = all_pred[2]
                    # left_mid_pred = all_pred[3]

                    all_right_sup_pred.append(target_scores[0])
                    all_left_sup_pred.append(target_scores[1])
                    all_right_mid_pred.append(target_scores[2])
                    all_left_mid_pred.append(target_scores[3])

                    all_right_sup_target.append(all_pred[0])
                    all_left_sup_target.append(all_pred[1])
                    all_right_mid_target.append(all_pred[2])
                    all_left_mid_target.append(all_pred[3])

                    # print(all_pred)
                    # print(target_scores.shape)

                    pbar.update()
            return all_right_sup_pred , all_left_sup_pred , all_right_mid_pred , all_left_mid_pred, all_right_sup_target , all_left_sup_target , all_right_mid_target , all_left_mid_target

    def train_and_validation(self, batch_train_data, batch_val_data, all_class_weights):

        logging.info("Training...")
        loss_avg = RunningAverage()
        self.model.train()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        with tqdm(total=len(batch_train_data)) as pbar:
                
            nan_val = 0
            for i, samples_batch in enumerate(batch_train_data): 

                input_img = Variable(samples_batch[0]).to(self.device)
                target_scores = samples_batch[1].type(torch.LongTensor)
                target_scores = Variable(target_scores).to(self.device)
                # target_scores = Variable(samples_batch[1]).to(self.device)

                if torch.isnan(input_img).any():
                    nan_val=+1
                    continue
                
                # pred_1, pred_2, pred_3, pred_4, pred_5, pred_6 = self.model(input_img)
                pred_1, pred_2, pred_3, pred_4 = self.model(input_img)
                # all_pred = [pred_1, pred_2, pred_3, pred_4, pred_5, pred_6]
                all_pred = [pred_1, pred_2, pred_3, pred_4]

                all_loss_fn = self._select_criterion(all_class_weights)

                # loss = all_loss_fn[1](all_pred[1], target_scores[:,1])
                loss = sum([all_loss_fn[i](all_pred[i], target_scores[:,i]) for i in range(4)])

                if self.args.use_amp:
                    # back propagration with scaled loss
                    scaler.scale(loss).backward(retain_graph=False)

                    # Unscales the gradients and clip them
                    scaler.unscale_(self.model_optim)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)

                    # optimizer step
                    scaler.step(self.model_optim)

                    # clear gradients
                    self.model_optim.zero_grad()

                    #update the scale for next iteration
                    scaler.update()
                else:
                    # back propagration : calculating the gradients
                    loss.backward(retain_graph=False) 

                    # Update model parameters
                    self.model_optim.step() 
                
                    # Zeo gradient
                    self.model_optim.zero_grad()

                # progress bar
                loss_avg.update(loss.item()) 
                pbar.set_postfix(loss='{:05.3f}'.format(loss_avg()))
                pbar.update()

            logging.info("- number of files used for training which contain NaN value : {}".format(nan_val))
            logging.info("Loss parameters: {}".format(list(all_loss_fn[0].parameters())))
            logging.info("Loss train: {:05.3f}".format(loss_avg()))

        train_loss = loss_avg()

        validation_loss = self.validation(self.model, batch_val_data, all_loss_fn)

        return train_loss, validation_loss, self.model_optim, self.model, all_loss_fn

        


