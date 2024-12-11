import os
import sys

import pandas as pd
from tqdm import tqdm
from torch.autograd import Variable

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
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--num_epochs', type=int, default=30, help='number of epochs') 
parser.add_argument('--use_amp', type=bool, default=False, help='use automatic mixed precision training')

args = parser.parse_args()

if __name__ == '__main__':

    # Select device and set the random seed for reproducible experiments
    device = args.device
    torch.manual_seed(123)

    # create directory to save if it does not exist
    experiment_path = f'{args.experiment_path}{args.model}_1'
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    #Set the logger
    utils.set_logger(os.path.join(experiment_path, 'train.log'))

    logging.info('loading dataset...')
    # batch_train_data, batch_val_data, batch_test_data = get_data(args.batch_size)
    batch_size = 16
    batch_train_data, batch_val_data, batch_test_data = get_data(batch_size)

    logging.info('Instantiating the model ({})...'.format(args.model))
    main_exp = Exp_Main(args)

    logging.info("Open tensorboard writer")
    writer = SummaryWriter(experiment_path + '/runs/' + args.model) 

    logging.info("Starting with... {} model".format(args.model))

    # train = False
    train = True

    if train == True:
        best_loss = 10
        befor_epoch_time = time.time()
        for epoch in range(args.num_epochs):

            # Run one epoch
            logging.info("Epoch {}/{}".format(epoch + 1, args.num_epochs))

            logging.info("Training and Validating...")
            train_loss, val_loss, train_accuracy, val_accuracy = main_exp\
                .train_and_validation(batch_train_data, batch_val_data)
                
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
            
            # If best_eval, best_save_path
            if is_best:
                logging.info("- Found new best scrore")
                best_loss = val_loss 

        logging.info("End, epoch: {}/{}. cost time in seconds: {}".format(epoch + 1, args.num_epochs, time.time() - befor_epoch_time))

    else:
        logging.info("Starting evaluation with {}...".format(args.model))

        model_to_load = main_exp.model
        utils.load_checkpoint(os.path.join(experiment_path, 'best.pth.tar'), model_to_load)

        file_to_save = f'{experiment_path}/results'
        if not os.path.exists(file_to_save): 
            os.makedirs(file_to_save)

        ########## tes model performances #############
        main_exp.get_performance_per_class(batch_val_data, model_to_load, device)
            
        # file = 'experiments/Rad_Dino_1/runs/Rad_Dino/'
        # utils.convert_tensorbord_to_csv(file, file_to_save)
        # csv_file = 'experiments/cheXNet_1/results/loss_curve.csv'
        # png_file = 'experiments/cheXNet_1/results/loss_function.png'
        # utils.plot_tensorbord(csv_file, png_file, 'cheXNet_1', 'CE')


    # # dic = {}
    # # for name, data in zip(['val_loss','test_loss' ], [batch_val_data, batch_test_data]):
    # #     loss_fn =main_exp._select_criterion(all_class_weights)
    # #     loss_val =  main_exp.validation(model_to_load, data, loss_fn)

    # #     dic[name]= loss_val.item()

    # # save_path = os.path.join(file_to_save, "value_of_loss_.json")
    # # utils.save_dict_to_json(dic, save_path)
        

    # data_f = pd.DataFrame()
    # for name, data in zip(['test_loss' ], [batch_test_data]):
    #     loss_fn =main_exp._select_criterion(all_class_weights)
    #     all_right_sup_pred , all_left_sup_pred , all_right_mid_pred , all_left_mid_pred, all_right_sup_target , all_left_sup_target , all_right_mid_target , all_left_mid_target  =  main_exp.validation(model_to_load, data, loss_fn)

    #     print(all_right_sup_pred) 
    #     print()
    #     print(all_left_sup_pred)

    #     # dic[name]= loss_val.item()

    # data_f['all_right_sup_pred'] = [t.cpu().tolist() for t in torch.cat(all_right_sup_pred)]
    # data_f['all_right_sup_target'] = [t.cpu().tolist() for t in torch.cat(all_right_sup_target)]

    # data_f['all_left_sup_pred'] = [t.cpu().tolist() for t in torch.cat(all_left_sup_pred)]
    # data_f['all_left_sup_target'] = [t.cpu().tolist() for t in torch.cat(all_left_sup_target)]

    # data_f['all_right_mid_pred'] = [t.cpu().tolist() for t in torch.cat(all_right_mid_pred)]
    # data_f['all_right_mid_target'] = [t.cpu().tolist() for t in torch.cat(all_right_mid_target)]

    # data_f['all_left_mid_pred'] = [t.cpu().tolist() for t in torch.cat(all_left_mid_pred)]
    # data_f['all_left_mid_target'] = [t.cpu().tolist() for t in torch.cat(all_left_mid_target)]

    # data_f.to_csv('yoo.csv')

    
    # ## get scores ##
    # # dir_to_save_scores = f'{file_to_save}/scores'
    # # if not os.path.exists(dir_to_save_scores):
    # #     os.makedirs(dir_to_save_scores)
    # # utils.save_scores(args, batch_test_data, model_to_load, device, dir_to_save_scores)

    # # init_file = os.path.join(experiment_path, 'runs', args.model )
    # # csv_file = f'{file_to_save}/loss_function/loss_curve.csv'
    # # png_file = f'{file_to_save}/loss_function/loss_fn'

    # # utils.convert_tensorbord_to_csv(init_file, f'{file_to_save}/loss_function')
    # # utils.plot_tensorbord(csv_file, png_file, 'Train & Val curves', 'MSE')

    # df = pd.read_csv(f'{file_to_save}/scores/scores_images.csv')

    # data_list = [df.diff_right_sup, df.diff_left_sup, df.diff_right_mid, df.diff_left_mid]
    # data_name = ['right_sup', 'left_sup', 'right_mid', 'left_mid']

    # utils.plot_box_plot(data_list, data_name, f'{file_to_save}/fn_save')