import os
import sys

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from exp.exp_main import Exp_Main

current_path =  os.path.abspath(__file__) 
project_root = os.path.abspath(os.path.join(current_path, '..')) 
sys.path.append(project_root)

import utils
# from exp.exp_main import Exp_Main
from data_provider.dataloader import get_data

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import time
import torch
import logging
import argparse
from torch.utils.tensorboard import SummaryWriter

# parser = argparse.ArgumentParser(description='NN for scoring fragments of X ray images')
parser = argparse.ArgumentParser(description='NN for classifying X ray images')

# basic config
parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

parser.add_argument('--experiment_path', type=str, default='experiments/', help='path to save the weight of the model')
parser.add_argument('--model', type=str, default='Rad_Dino',
                    help='model name, options: [Rad_Dino, cheXNet, ResNet, initial_vit]')

# data loader
parser.add_argument('--data_dirr', type=str, default='data_1', help='directory of data')

# optimization 
parser.add_argument('--learning_rate', type=float, default=1e-5, help='optimizer learning rate') # 
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--num_epochs', type=int, default=30, help='number of epochs') 
parser.add_argument('--use_amp', type=bool, default=False, help='use automatic mixed precision training')

args = parser.parse_args()

if __name__ == '__main__':

    # Select device and set the random seed for reproducible experiments
    device = args.device
    torch.manual_seed(123)

    # create directory to save if it does not exist
    args.model = 'Rad_Dino'
    experiment_path = f'{args.experiment_path}{args.model}'
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    #Set the logger
    utils.set_logger(os.path.join(experiment_path, 'train.log'))

    logging.info('loading dataset...')
    batch_train_data, batch_val_data, batch_test_data = get_data(args.batch_size)

    logging.info('Instantiating the model ({})...'.format(args.model))
    main_exp = Exp_Main(args)

    logging.info("Starting with... {} model".format(args.model))

    # train = False
    train = True

    best_loss = 10
    befor_epoch_time = time.time()
    for epoch in range(args.num_epochs):

        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, args.num_epochs))

        logging.info("Training and Validating...")
        train_loss, val_loss, train_accuracy, val_accuracy = main_exp\
            .train_and_validation(batch_train_data, batch_val_data)
            
        logging.info("Open tensorboard writer")
        writer = SummaryWriter(experiment_path + '/runs/' + args.model) 

        # Write loss and metric on tensorboard
        writer.add_scalar('/loss-train', train_loss, epoch +1) 
        writer.add_scalar('/loss-val', val_loss, epoch + 1) 

        writer.add_scalar('/train-accuracy', train_accuracy, epoch + 1) 
        writer.add_scalar('/val-accuracy', val_accuracy, epoch + 1) 

        is_best = val_loss <= best_loss
        
        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                                'model': main_exp.model.state_dict(),
                                'optim_dict': main_exp.model_optim.state_dict()},
                                is_best=is_best,
                                checkpoint=experiment_path)
        
        # Update the new best loss
        if is_best:
            logging.info("- Found new best scrore")
            best_loss = val_loss 

    logging.info("End, epoch: {}/{}. cost time in seconds: {}".format(epoch + 1, args.num_epochs, time.time() - befor_epoch_time))
