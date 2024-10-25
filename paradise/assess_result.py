

import os
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, cohen_kappa_score
import seaborn as sns
import matplotlib.pyplot as plt
import utils


data_df = pd.read_csv('wkdir/result/rad_dino_CE.csv')

# ,all_right_sup_pred,all_right_sup_target,
# all_left_sup_pred,all_left_sup_target,
# all_right_mid_pred,all_right_mid_target,
# all_left_mid_pred,all_left_mid_target


targets = data_df.all_left_mid_pred
predicted_classes = data_df.all_left_mid_target

precision, recall, f1, _ = precision_recall_fscore_support(targets, predicted_classes, average='weighted')
accuracy = (predicted_classes == targets).sum().item() / len(targets)
precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(targets, predicted_classes, average=None)
cm = confusion_matrix(targets, predicted_classes)
kappa = cohen_kappa_score(targets, predicted_classes)


print(f'Precision per class: {precision_per_class}')
print(f'F1-Score per class: {f1_per_class}')
print(f'Recall per class: {recall_per_class}')

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')

print('accuracy: ', accuracy)

print(f"Cohen's Kappa: {kappa:.4f}")

dic = {}

dic['precision_per_class']= list(precision_per_class)
dic['recall_per_class']= list(recall_per_class)
dic['f1_per_class']= list(f1_per_class)

dic['precision']= precision
dic['recall']= recall
dic['f1']= f1

dic['accuracy']= accuracy
dic["Cohen's Kappa"]= kappa

dirr = 'wkdir/result/rad_dino_CE/left_mid'

if not os.path.exists(dirr):
        os.makedirs(dirr)

save_path = os.path.join(dirr, 'metrics.json')
utils.save_dict_to_json(dic, save_path)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(f'{dirr}/cm.png')


