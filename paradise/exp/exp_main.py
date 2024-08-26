import logging

import numpy as np

# import utils
# from utils.metrics import metric
from models.vit import CustomDinoModel, model 
from models.resnet import resnet34
from utils import RunningAverage

import torch

import torch.nn as nn
from tqdm import tqdm
from torch import optim
from torch.autograd import Variable

# from torch.nn import functional as F

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
            'ResNet': resnet34()
        }

        if self.args.model == 'ResNet':
            self.model = model_dict[self.args.model].to(self.device)
        else: 
            self.model = model_dict[self.args.model](model).to(self.device)
        
        self.model_optim = self._select_optimizer()

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def run_model(self, model, x):
        outputs = model(x)
        return outputs

    def validation(self, model , vali_loader, loss_fn):
        print()
        logging.info("Validation...")
        total_loss = []
        model.eval()
        with torch.no_grad():
            nan_val = 0
            with tqdm(total=len(vali_loader)) as pbar:
                    
                for samples_batch in vali_loader:
                    
                    input_img = Variable(samples_batch[0]).to(self.device)
                    target_scores = Variable(samples_batch[1]).to(self.device)

                    if torch.isnan(input_img).any():
                        nan_val=+1
                        continue
                    output_model = model(input_img)
                    loss = loss_fn(output_model, target_scores)

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

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        with tqdm(total=len(batch_train_data)) as pbar:
                
            nan_val = 0
            for i, samples_batch in enumerate(batch_train_data): 

                input_sig = Variable(samples_batch[0]).to(self.device)
                target_sig = Variable(samples_batch[1]).to(self.device)

                if torch.isnan(input_sig).any():
                    nan_val=+1
                    continue
                
                output_batch_all_lead = self.model(input_sig)

                loss_fn = self._select_criterion()
                loss = loss_fn(output_batch_all_lead, target_sig)

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
            logging.info("Loss parameters: {}".format(list(loss_fn.parameters())))
            logging.info("Loss train: {:05.3f}".format(loss_avg()))

        train_loss = loss_avg()

        validation_loss = self.validation(self.model, batch_val_data, loss_fn)

        return train_loss, validation_loss, self.model_optim, self.model, loss_fn

        


