import json
import logging
import shutil
import traceback
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


from sklearn.metrics import confusion_matrix, balanced_accuracy_score, precision_recall_fscore_support

import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import torch
from tqdm import tqdm

def drop_nan_df(input_dir = 'data_1/labels/paradise_csi', output_dir = 'data_1/labels/paradise_csi_drop_nan'):

    df = pd.read_csv(f'{input_dir}.csv')
    # df = pd.DataFrame(data=df, columns=['number','id_number', 'csi_total','csi', 'right_sup', 
    #                                                         'left_sup','right_mid',
    #                                                         'left_mid','right_mid',])
    
    missing_values = df.isnull().sum()
    # print(missing_values[missing_values > 0])  # Columns with missing values
    df_cleaned = df.dropna()
    
    df_cleaned.to_csv(f'{output_dir}.csv')

def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)
    
def plot_tensorbord(csv_file, png_file, title, y_ax):
    data = pd.read_csv(csv_file)

    loss_train = list((data[data.metric.isin(['/loss-train'])].value))
    loss_val =  list((data[data.metric.isin(['/loss-val'])].value))

    min_train = '{:05.3f}'.format(min(loss_train))
    min_val = '{:05.3f}'.format(min(loss_val))

    plt.figure()
    plt.title(f'{title} (val : {min_val}, train : {min_train})')
    plt.plot(loss_train, label='Training')  
    plt.plot(loss_val, label='Validation')  
    plt.xlabel('Epochs')
    plt.ylabel(y_ax)
    plt.legend()
    plt.savefig(png_file)
    plt.close()

def convert_tensorbord_to_csv(folderpath, folder_to_save):

    # Extraction function
    def tflog2pandas(path):
        runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
        try:
            event_acc = EventAccumulator(path)   
            event_acc.Reload()
            tags = event_acc.Tags()["scalars"]
            for tag in tags:
                event_list = event_acc.Scalars(tag)
                values = list(map(lambda x: x.value, event_list))
                step = list(map(lambda x: x.step, event_list))
                r = {"metric": [tag] * len(step), "value": values, "step": step}
                r = pd.DataFrame(r)

                runlog_data = pd.concat([runlog_data, r])

        # Dirty catch of DataLossError
        except Exception:
            print("Event file possibly corrupt: {}".format(path))
            traceback.print_exc()
        return runlog_data
    path=folderpath #folderpath
    print(path)
    df=tflog2pandas(path)
    df.to_csv(f'{folder_to_save}/loss_curve.csv')

def load_checkpoint(checkpoint_path, model, optimizer=None):

    """Loads model parameters (model) from checkpoint_path. If optimizer is provided, loads model of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint_path: (string) file directory of the checkpoint which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint_path):
        raise("File doesn't exist {}".format(checkpoint_path))
    
    else:
        print()
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])

        if optimizer:
            optimizer.load_state_dict(checkpoint['optim_dict'])

def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory")
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists!")
        
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))

def save_dict_to_json(data, json_path):
    """Saves dict of floats in json file
    Args:
        data: dict
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as file:
        json.dump(data, file, indent=4)

def save_scores(args, test_data, model, device, dir_to_save_scores):

    dir_to_save_scores = dir_to_save_scores

    if not os.path.exists(dir_to_save_scores):
        os.mkdir(dir_to_save_scores)

    logging.info("Name of the directory to save scores' image : {}".format(dir_to_save_scores))

    print()

    others_fnames = []

    logging.info(f'Extracting scores from {len(test_data)*args.batch_size} x ray images.')

    pred_ref_scores_and_fnames = []
    with tqdm(total = len(test_data)) as pbar:
        for samples_batch in test_data:
    
            total_scores = test_data.__len__()

            input_img = samples_batch[0].to(device)
            ref_scores = samples_batch[1][0]
            fname = samples_batch[2][0].split('/')[-2]

            output_model = model(input_img)

            try:
                output_model = model(input_img).squeeze(0)
                ref_scores = ref_scores.tolist()
                the_fname = fname

                output_model = [round(val, 2) for val in output_model.tolist()]

                dic = {'predicted_scores': output_model, 'reference_scores': ref_scores, 'fname': the_fname}
                pred_ref_scores_and_fnames.append(dic)

            except AssertionError as e:
                print(f'AssertionError : {e}')
            pbar.update()

        df = pd.DataFrame()
        for i in range(0,4):
            predicted_scores = [dic['predicted_scores'][i] for dic in pred_ref_scores_and_fnames]
            reference_scores = [dic['reference_scores'][i] for dic in pred_ref_scores_and_fnames ]
            fname = [dic['fname'] for dic in pred_ref_scores_and_fnames]

            diff = [round(val, 3) for val in  stat_difference(predicted_scores, reference_scores)]

            if i ==0:
                df['fname']=fname  
                df['right_sup_pred'] = predicted_scores
                df['right_sup_ref'] = reference_scores
                df["diff_right_sup"] = diff
            if i ==1:
                df['left_sup_pred'] = predicted_scores
                df['left_sup_ref'] = reference_scores
                df["diff_left_sup"] = diff
            elif i==2:
                df['right_mid_pred'] = predicted_scores
                df['right_mid_ref'] = reference_scores
                df["diff_right_mid"] = diff
            elif i == 3:
                df['left_mid_pred'] = predicted_scores
                df['left_mid_ref'] = reference_scores
                df["diff_left_mid"] = diff

        df.to_csv(f"{dir_to_save_scores}/scores_images.csv")
        logging.info("Done... {}/{} scores extracted".format(len(pred_ref_scores_and_fnames), total_scores))
        print()
        
    file = pd.DataFrame()  
    file['fname'] = [dic['fname'] for dic in others_fnames]
    file['predicted_mask'] = [dic['predicted_mask'] for dic in others_fnames]
    file['reference_mask'] = [dic['reference_mask'] for dic in others_fnames]

    file.to_csv('{}/not_usable_fnames.csv'.format(dir_to_save_scores))

def stat_difference(actual, predicted):
    diffs = [val1 - val2 for val1, val2 in zip(actual, predicted)]
    return diffs

def plot_box_plot(data_list, data_name, fn_save):

    plt.figure()
    plt.title("ResNet's predictions vs automatics scores on test set")
    plt.boxplot(data_list,  notch = True, showfliers=True)
    # plt.boxplot(data_list,  notch = True, showfliers=False)
    plt.xticks(np.arange(len(data_name))+1,data_name)
    plt.ylabel('manual vs automatic')
    plt.savefig(f'{fn_save}_box_plot.png')

def conf_matrix(targets, predicted_classes):

    cm = confusion_matrix(targets, predicted_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def add_class_for_scores():

    paradise_csi = pd.read_csv('data_1/labels/paradise_csi.csv')
    print(paradise_csi.columns)

    df = pd.DataFrame(paradise_csi)

    # Add the new column based on conditions
    def classify_csi(csi):
        if csi <= 1.3:
            return 'AHF_risk = 4.4%'
        elif 1.3 < csi <= 2.2:
            return 'AHF_risk = 25.8%'
        else:
            return 'AHF_risk = 60.3%'

    df['classes'] = df['csi'].apply(classify_csi)
    df['classes_label'] = df['classes'].map({'AHF_risk = 4.4%':0, 'AHF_risk = 25.8%':1, 'AHF_risk = 60.3%':2})
    df.to_csv('data_1/labels/paradise_csi_w_classes.csv')

    print(df['classes_label'].value_counts())

def replace_nan(file_csv):
    paradise_csi = pd.read_csv(file_csv)
    df = pd.DataFrame(paradise_csi)
    df.replace(r'^\s*$', float('nan'), regex=True)
    df.fillna(0, inplace=True)
    df.to_csv('yep_.csv')