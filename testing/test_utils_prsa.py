from os.path import join
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
import json
import transformers
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from configs.config_prsa import CONFIG_DICT, CONFIG_FINETUNING
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def test_prsa_model(model, test_dataset, output_dir):
    # logger.debug("\n")
    # batch_size = 1  # TODO: da commentare (SMALL)
    batch_size = CONFIG_FINETUNING['batch_size']
    num_columns = len(test_dataset.dataset.input_vocab.get_field_keys(remove_target=True, ignore_special=True))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=transformers.DefaultDataCollator())
    model.eval()

    scaler_pm10 = test_dataset.dataset.encoding_fn['PM10']
    scaler_pm25 = test_dataset.dataset.encoding_fn['PM2.5']

    all_losses = []
    all_preds = np.empty((0,CONFIG_FINETUNING['seq_len'],2))
    all_labels = np.empty((0,CONFIG_FINETUNING['seq_len'],2))
    
    for input in tqdm(test_dataloader):

        input_ids = input['input_ids'].to(device)
        labels = input['labels'].to(device)

        output = model(input_ids=input_ids, labels=labels)

        loss, preds = output

        all_losses.append(loss.cpu().numpy())
        # logger.info(f'all_preds shape: {all_preds.shape}, preds shape: {preds.shape}')
        # logger.info(f'all_labels shape: {all_labels.shape}, labels shape: {labels.shape}')
        all_preds = np.concatenate((all_preds, preds.cpu().numpy()), axis=0)
        all_labels = np.concatenate((all_labels, labels.cpu().numpy()), axis=0)

    assert all_preds.shape == all_labels.shape, f'probs shape {all_preds.shape} is not equal to labels shape {all_labels.shape}'
    # scale all preds to original values
    all_preds[:,:,0] = scaler_pm25.inverse_transform(all_preds[:,:,0])
    all_preds[:,:,1] = scaler_pm10.inverse_transform(all_preds[:,:,1])
    all_labels[:,:,0] = scaler_pm25.inverse_transform(all_labels[:,:,0])
    all_labels[:,:,1] = scaler_pm10.inverse_transform(all_labels[:,:,1])
    # Statistics
    print_statistics(all_preds, all_labels, output_dir)


@torch.inference_mode()
def test_prsa_model_new(model, test_dataset, output_dir, datacollator):
    # logger.debug("\n")
    # batch_size = 1  # TODO: da commentare (SMALL)
    batch_size = CONFIG_FINETUNING['batch_size']
    num_columns = len(test_dataset.dataset.input_vocab.get_field_keys(remove_target=True, ignore_special=True))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=datacollator)
    model.eval()

    scaler_pm10 = test_dataset.dataset.encoding_fn['PM10']
    scaler_pm25 = test_dataset.dataset.encoding_fn['PM2.5']

    all_preds = np.empty((0,CONFIG_FINETUNING['seq_len'],2))
    all_labels = np.empty((0,CONFIG_FINETUNING['seq_len'],2))
    
    for input in tqdm(test_dataloader):

        input_ids = input['input_ids'].to(device)
        labels = input['labels'].to(device)

        output = model(input_ids=input_ids)
        preds = output[0]

        # logger.info(f'all_preds shape: {all_preds.shape}, preds shape: {preds.shape}')
        # logger.info(f'all_labels shape: {all_labels.shape}, labels shape: {labels.shape}')
        all_preds = np.concatenate((all_preds, preds.cpu().numpy()), axis=0)
        all_labels = np.concatenate((all_labels, labels.cpu().numpy()), axis=0)

    assert all_preds.shape == all_labels.shape, f'probs shape {all_preds.shape} is not equal to labels shape {all_labels.shape}'
    # scale all preds to original values
    all_preds[:,:,0] = scaler_pm25.inverse_transform(all_preds[:,:,0])
    all_preds[:,:,1] = scaler_pm10.inverse_transform(all_preds[:,:,1])
    all_labels[:,:,0] = scaler_pm25.inverse_transform(all_labels[:,:,0])
    all_labels[:,:,1] = scaler_pm10.inverse_transform(all_labels[:,:,1])
    # Statistics
    print_statistics(all_preds, all_labels, output_dir)


def print_statistics(preds, labels, output_dir):

    l = len(preds.shape)
    preds_pm25 = preds[...,0]
    preds_pm10 = preds[...,1]
    labels_pm25 = labels[...,0]
    labels_pm10 = labels[...,1]
    preds = preds.flatten()
    preds_pm10 = preds_pm10.flatten()
    preds_pm25 = preds_pm25.flatten()
    labels = labels.flatten()
    labels_pm10 = labels_pm10.flatten()
    labels_pm25 = labels_pm25.flatten()
    mae = mean_absolute_error(labels, preds)
    mae_perc = mean_absolute_percentage_error(labels, preds)
    mse = mean_squared_error(labels, preds)
    rmse = mean_squared_error(labels, preds, squared=False)

    mae_pm25 = mean_absolute_error(labels_pm25, preds_pm25)
    mae_perc_pm25 = mean_absolute_percentage_error(labels_pm25, preds_pm25)
    mse_pm25 = mean_squared_error(labels_pm25, preds_pm25)
    rmse_pm25 = mean_squared_error(labels_pm25, preds_pm25, squared=False)
    mae_pm10 = mean_absolute_error(labels_pm10, preds_pm10)
    mae_perc_pm10 = mean_absolute_percentage_error(labels_pm10, preds_pm10)
    mse_pm10 = mean_squared_error(labels_pm10, preds_pm10)
    rmse_pm10 = mean_squared_error(labels_pm10, preds_pm10, squared=False)

    count_labels_tot = labels.shape[0]
    count_labels_pm25 = labels_pm25.shape[0]
    count_labels_pm10 = labels_pm10.shape[0]
    count_preds_tot = preds.shape[0]
    count_preds_pm25 = preds_pm25.shape[0]
    count_preds_pm10 = preds_pm10.shape[0]
    mean_labels_tot = np.mean(labels)
    mean_labels_pm25 = np.mean(labels_pm25)
    mean_labels_pm10 = np.mean(labels_pm10)
    mean_preds_tot = np.mean(preds)
    mean_preds_pm25 = np.mean(preds_pm25)
    mean_preds_pm10 = np.mean(preds_pm10)

    stats_dict = {
        'Mean absolute error GLOBAL': mae,
        'Mean absolute percentage error GLOBAL': mae_perc,
        'Mean squared error GLOBAL': mse,
        'Root mean squared error GLOBAL': rmse,
        'Mean absolute error PM 2,5': mae_pm25,
        'Mean absolute percentage error PM 2,5': mae_perc_pm25,
        'Mean squared error PM 2,5': mse_pm25,
        'Root mean squared error PM 2,5': rmse_pm25,
        'Mean absolute error PM 10': mae_pm10,
        'Mean absolute percentage error PM 10': mae_perc_pm10,
        'Mean squared error PM 10': mse_pm10,
        'Root mean squared error PM 10': rmse_pm10,
        'count_labels_tot': count_labels_tot,
        'count_labels_pm25': count_labels_pm25,
        'count_labels_pm10': count_labels_pm10,
        'count_preds_tot': count_preds_tot,
        'count_preds_pm25': count_preds_pm25,
        'count_preds_pm10': count_preds_pm10,
        'mean_labels_tot': mean_labels_tot,
        'mean_labels_pm25': mean_labels_pm25,
        'mean_labels_pm10': mean_labels_pm10,
        'mean_preds_tot': mean_preds_tot,
        'mean_preds_pm25': mean_preds_pm25,
        'mean_preds_pm10': mean_preds_pm10,
    }
    logger.info(f'computed test stats: \n{json.dumps(stats_dict, indent=4)}')
    if l==2:
        with open(join(output_dir, 'main_statistics_mean.json'), 'w+') as fw:
            json.dump(stats_dict, fw, indent=4)
    else:
        with open(join(output_dir, 'main_statistics.json'), 'w+') as fw:
            json.dump(stats_dict, fw, indent=4)

