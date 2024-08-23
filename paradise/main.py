import os
import sys

from exp.exp_main import Exp_Main

current_path =  os.path.abspath(__file__) 
project_root = os.path.abspath(os.path.join(current_path, '..')) 
sys.path.append(project_root)

import utils
# from exp.exp_main import Exp_Main
from data_provider.dataloader import get_data

import time
import torch
import logging
import argparse
# from numpy import random
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='NN for scoring fragments of X ray images')

# basic config
parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

parser.add_argument('--experiment_path', type=str, default='experiments/', help='path to save the weight of the model')
parser.add_argument('--model', type=str, default='Rad_Dino',
                    help='model name, options: [Rad_Dino, Initial_ViT]')

# data loader
parser.add_argument('--data_dirr', type=str, default='data_2/CinC2021Challenge/full_set_v2', help='directory of data 2')

# optimization 
parser.add_argument('--learning_rate', type=float, default=1e-5, help='optimizer learning rate') # 
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs') 
parser.add_argument('--use_amp', type=bool, default=False, help='use automatic mixed precision training')

args = parser.parse_args()

if __name__ == '__main__':

    experiment_path = f'{args.experiment_path}{args.model}'
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    # Select device and set the random seed for reproducible experiments
    device = args.device
    torch.manual_seed(123)

    #Set the logger
    utils.set_logger(os.path.join(experiment_path, 'train.log'))

    logging.info('loading dataset...')
    batch_train_data, batch_val_data, batch_test_data = get_data(args.batch_size)

    logging.info('Instantiating the model ({})...'.format(args.model))
    main_exp = Exp_Main(args)


    logging.info("Starting... {}".format(args.model))
    logging.info("Open tensorboard writer")
    writer = SummaryWriter(experiment_path + '/runs/' + args.model) 

    best_loss = 10
    befor_epoch_time = time.time()
    for epoch in range(2):

        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, args.num_epochs))

        logging.info("Training and Validating...")
        train_loss, val_loss, model_optimizer, model, loss_fn = main_exp.train_and_validation(batch_train_data, batch_val_data)
            
        # Write loss and metric on tensorboard
        writer.add_scalar('/loss-train', train_loss, epoch +1) 
        writer.add_scalar('/loss-val', val_loss, epoch + 1) 

        is_best = val_loss <= best_loss
        
        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                                'model': model.state_dict(),
                                'optim_dict': model_optimizer.state_dict()},
                                is_best=is_best,
                                checkpoint=experiment_path)
        
        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best scrore")
            best_loss = val_loss 

    logging.info("Epoch: {} cost time in seconds: {}".format(epoch + 1, time.time() - befor_epoch_time))

    logging.info("Starting evaluation with {}...".format(args.model))

    model_to_load = main_exp.model
    utils.load_checkpoint(os.path.join(experiment_path, 'best.pth.tar'), model_to_load)

    file_to_save = f'{experiment_path}/results'
    if not os.path.exists(file_to_save):
        os.makedirs(file_to_save)

    dic = {}
    for name, data in zip(['val_loss','test_loss' ], [batch_val_data, batch_test_data]):
        loss_fn =main_exp._select_criterion()
        loss_val =  main_exp.validation(main_exp.model, data, loss_fn)

        dic[name]= loss_val.item()

    save_path = os.path.join(file_to_save, "value_of_loss_.json")
    utils.save_dict_to_json(dic, save_path)