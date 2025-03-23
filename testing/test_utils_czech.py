from genericpath import isfile
import os
import re
from os import makedirs
from os.path import join
from unittest import result
import numpy as np
import torch
import random
from loguru import logger
from tqdm import tqdm
import json
import pandas as pd
import math
from datetime import datetime
from scipy import stats
import transformers
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

from configs.config_card import CONFIG_DICT, CONFIG_FINETUNING
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def test_czech_model(model, test_dataset, output_dir, numero_split=""):
    # logger.debug("\n")
    # batch_size = 1  # TODO: da commentare (SMALL)
    batch_size = CONFIG_FINETUNING['batch_size']
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=transformers.DefaultDataCollator())
    model.eval()

    all_losses = []
    all_probs = np.array((0,))
    all_labels = np.array((0,))
    
    for input in tqdm(test_dataloader):

        input_ids = input['input_ids'].to(device)
        labels = input['labels'].to(device)

        output = model(input_ids=input_ids, labels=labels)

        loss, logits = output
        labels = labels
        probs = logits.sigmoid()

        all_losses.append(loss.cpu().numpy())
        all_probs = np.concatenate((all_probs, probs.cpu().numpy()))
        all_labels = np.concatenate((all_labels, labels.cpu().numpy()))

    assert all_probs.shape == all_labels.shape, f'probs shape {probs.shape} is not equal to labels shape {labels.shape}'
    # Statistics
    stats_dict = print_statistics(all_probs, all_labels, output_dir, numero_split)
    # print_model_stats(model, output_dir, num_columns)
    return stats_dict


@torch.inference_mode()
def test_czech_model_new(model, test_dataset, data_collator, output_dir, numero_split=""):
    # logger.debug("\n")
    # batch_size = 1  # TODO: da commentare (SMALL)
    batch_size = CONFIG_FINETUNING['batch_size']
    # num_columns = test_dataset.dataset.ncols
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=data_collator)
    model.eval()

    all_probs = np.array((0,))
    all_labels = np.array((0,))
    
    for input in tqdm(test_dataloader):

        input_ids = input['input_ids'].to(device)
        labels = input['labels'].to(device)

        output = model(input_ids=input_ids)
        logits = output[0]

        probs = logits.sigmoid()

        all_probs = np.concatenate((all_probs, probs.cpu().numpy()))
        all_labels = np.concatenate((all_labels, labels.cpu().numpy()))

    assert all_probs.shape == all_labels.shape, f'probs shape {probs.shape} is not equal to labels shape {labels.shape}'
    # Statistics
    stats_dict = print_statistics(all_probs, all_labels, output_dir, numero_split)
    # print_model_stats(model, output_dir, num_columns)
    return stats_dict


def binary_results_from_predictions(preds, labels):

    confusion_mat = confusion_matrix(labels, preds)
    tp, fp = int(confusion_mat[1,1]), int(confusion_mat[0,1])
    tn, fn = int(confusion_mat[0,0]), int(confusion_mat[1,0])

    accuracy = accuracy_score(labels, preds)
    accuracy_mean = balanced_accuracy_score(labels, preds)
    f1_class1 = f1_score(labels, preds, average='binary')
    f1_macro = f1_score(labels, preds, average='macro')
    f1_weighted = f1_score(labels, preds, average='weighted')

    return {
        'accuracy': accuracy,
        'accuracy_mean': accuracy_mean,
        'confusion_matrix': {'tp':tp, 'fp': fp, 'fn': fn, 'tn': tn},
        'f1_target': f1_class1,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
    }


def print_statistics(probs, labels, output_dir, numero_split=""):

    ### FIRST COMPUTE RESULTS NOT DEPENDENT FROM THRESHOLD 
    ### THEN TRY DIFFERENT THRESHOLDING POLICIES
    total_true = int(np.sum(labels==1))
    total_false = int(np.sum(labels==0))

    # COMPUTE ROC Area-Under-Curve
    roc_auc = roc_auc_score(labels, probs)
    # COMPUTE Precision-Recall Area-Under-Curve
    PR_auc_classes = average_precision_score(labels, probs, average=None)
    PR_auc_macro = average_precision_score(labels, probs, average='macro')

    # Consider banal threshold of 0.5
    preds_05 = np.array(probs>0.5).astype(np.int32)
    results_05 = binary_results_from_predictions(preds_05, labels)
    logger.info(f'\nCOMPUTED RESULTS WITH thr=0.5 : \n{json.dumps(results_05, indent=4)}')

    # estimate threshold from precision recall curve maximizing F1-Score
    precision_list, recall_list, thresholds_list = precision_recall_curve(labels, probs)
    assert min(thresholds_list)>=0 and max(thresholds_list)<=1, f'the thresholds are not in [0,1] but in [{min(thresholds_list)},{max(thresholds_list)}], \n->{thresholds_list}'
    f1_list = (2*precision_list*recall_list)/(precision_list+recall_list)
    index1 = np.nanargmax(f1_list)
    threshold_f1 = thresholds_list[index1]
    preds_f1 = np.array(probs>threshold_f1).astype(np.int32)
    results_f1 = binary_results_from_predictions(preds_f1, labels)
    logger.info(f'\nCOMPUTED RESULTS WITH thr={round(threshold_f1,2)}, MAXIMIZING F1-SCORE : \n{json.dumps(results_f1, indent=4)}')

    # estimate threshold from roc curve maximizing G-mean
    fpr, tpr, thresholds_roc = roc_curve(labels, probs)
    gmeans = np.sqrt(tpr * (1-fpr))
    index2 = np.nanargmax(gmeans)
    threshold_roc = thresholds_roc[index2]
    preds_roc = np.array(probs>threshold_roc).astype(np.int32)
    results_roc = binary_results_from_predictions(preds_roc, labels)
    logger.info(f'\nCOMPUTED RESULTS WITH thr={round(threshold_roc,2)}, MAXIMIZING G-MEAN : \n{json.dumps(results_roc, indent=4)}')

    # PLOT PRECISION RECALL CURVE WITH AVERAGE PRECISION AND F1 SCORE
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig_img_rocauc = ax
    fig_img_rocauc.plot(recall_list, precision_list, label='precision recall curve')
    fig_img_rocauc.plot(recall_list, f1_list, label='precision recall f1 score')
    fig_img_rocauc.scatter(recall_list[index1], precision_list[index1], c='r', label='optimal threshold')
    fig_img_rocauc.title.set_text('Image precision recall, AP-score: {0:.2f} F1-score: {1:.2f}%, Optimal Threshold: {2:.2f}'.format(PR_auc_macro, f1_list[index1], threshold_f1))
    fig_img_rocauc.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'prec-rec_curve.png'), dpi=100)

    # PLOT ROC CURVE WITH ROC-AUC AND G-MEAN SCORE
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig_img_rocauc = ax
    fig_img_rocauc.plot(fpr, tpr, label='img_ROCAUC:{0:.3f}'.format(roc_auc))
    fig_img_rocauc.scatter(fpr[index2], tpr[index2], c='r', label='optimal threshold')
    fig_img_rocauc.title.set_text('Image ROCAUC:{0:.3f}, G-MEAN: {1:.1f}%, Optimal Threshold: {2:.2f}'.format(roc_auc, gmeans[index2], threshold_roc))
    fig_img_rocauc.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=100)

    stats_dict = {
        'total_true_samples':total_true,
        'total_false_sample':total_false,
        'AUC': roc_auc,
        'Average-Precisions': PR_auc_classes,
        'Average_precision_macro': PR_auc_macro,
        'Results thr=0,5': results_05,
        'Optimal thr F1': threshold_f1,
        'Results optimal F1': results_f1,
        'Optimal thr Gmean': threshold_roc,
        'Results optimal Gmean': results_roc,
    }
    with open(join(output_dir, 'main_statistics'+str(numero_split)+'.json'), 'w+') as fw:
        json.dump(stats_dict, fw, indent=4)

    return stats_dict


def print_model_stats(model, output_dir, num_columns):
    import time
    open(join(output_dir, f'model_stats.txt'), 'w+')

    batch_sizes = [1,10]
    for bs in batch_sizes:

        if CONFIG_FINETUNING['use_embeddings']:
            input_size = (bs, CONFIG_FINETUNING['seq_len']*num_columns, CONFIG_DICT['field_hidden_size'])
        else:
            input_size = (bs, CONFIG_FINETUNING['seq_len']*num_columns)

        input_ids = torch.rand(input_size).to(device)
        if not CONFIG_FINETUNING['use_embeddings']:
            input_ids = input_ids.to(torch.int64)

        with open(join(output_dir, f'model_stats.txt'), 'a') as fw:
            fw.write(f'STATS WITH BATCH SIZE = {bs}\n')

        pytorch_total_params = sum(p.numel() for p in model.parameters())
        pytorch_total_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2

        _ = model(input_ids=input_ids)
        start_time = time.perf_counter()
        for _ in range(100):
            _ = model(input_ids=input_ids)
        infer_time = time.perf_counter()-start_time
        with open(join(output_dir, f'model_stats.txt'), 'a') as fw:
            fw.write('pytorch model size = {:.3f}MB\n'.format(size_all_mb))
            fw.write(f'pytorch total params = {pytorch_total_params}\n')
            fw.write(f'pytorch total learnable params = {pytorch_total_learnable_params}\n')
            fw.write(f'inference time = {infer_time/100}\n')
            fw.write(f'inferences per second = {bs*100/infer_time}\n\n')
