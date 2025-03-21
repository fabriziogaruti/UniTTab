import os
from os import path
import pandas as pd
import numpy as np
import math
import json
import tqdm
import pickle
from loguru import logger
from sklearn.utils import resample
from random import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data.dataset import Dataset

from misc.utils import divide_chunks
from dataset.vocab import Vocabulary
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TransactionDataset(Dataset):
    def __init__(self,
                configs,
                root="/home/andrej/data/card/",
                fname="card_transaction_200",
                vocab_dir="risultati_training",
                return_labels=False,
                masked_lm=True,
                baseline=False,):

        self.tokenizer_pad_token_id = 1

        ## folder parameters and type of return are the only ones not in configs
        self.configs = configs
        self.root = root
        self.fname = fname
        self.vocab_dir = vocab_dir
        self.return_labels = return_labels
        self.mlm = masked_lm
        self.baseline = baseline

        self.input_vocab = Vocabulary(target_column_name='label')
        self.output_vocab = Vocabulary(target_column_name='label')
        self.encoding_fn = {}
        self.quantized_regression_values = {}
        self.quantized_regression_edges = {}

        logger.info(f'DATASET CONFIGURATION FILE: \n{json.dumps(configs, indent=4)}')
        # parameters to reduce the dataset
        self.nrows = self.configs['nrows']
        self.user_ids = self.configs['user_ids']

        self.skip_user = self.configs['skip_user']
        # parameters to decide how to handle negative amount and to include or not users
        self.segno_amount = self.configs['segno_amount']  # 0 non usa segno, 1 usa campo aggiuntivo, 2 somma il valore assoluto del min

        self.trans_stride = self.configs['stride']
        self.seq_len = self.configs['seq_len']
        self.columns_to_select = self.configs['columns_to_select']

        # parameters to decide how to handle continuous variables
        self.flatten = self.configs['flatten']
        self.regression = self.configs['regression']
        self.quantize_regression_output = self.configs['quantize_regression_output']
        # quantization parameters
        self.num_bins = self.configs['quantize_num_bins']
        self.numerical_num_bins = self.configs['numerical_num_bins']
        self.numerical_fields = self.configs['numerical_fields']  # todo: check se serve e check se init quant da fare subito?
        # regression parameters
        self.regression_fields = self.configs['regression_fields']
        self.normalize_regression = self.configs['normalize_regression']
        # combinatorial quantization parameters
        self.comb_fields = self.configs['combinatorial_quantization_fields']
        self.num_figures = self.configs['num_figures']
        self.base = self.configs['combinatorial_quantization_base']
        self.comma = self.configs['combinatorial_quantization_comma']

        # initialize some helper variables  # TODO: servono??
        self.ncols = None
        self.encoder_fit = {}  # dizionario contenente nome colonna e lista dei margini di quantizzazione
        self.trans_table = None

        self.data, self.labels = [], []  # = self.samples, self.targets in prsa
        self.indices = []

        # preprocess data create sequences and vocab
        self.encode_data()
        self.init_vocab()
        self.prepare_samples()
        self.save_vocab(self.vocab_dir)

    def __getitem__(self, index):
        real_index = self.indices[index]

        if self.regression:
            return_data = torch.tensor(self.data[real_index], dtype=torch.float).reshape(self.seq_len, -1)
        else:
            return_data = torch.tensor(self.data[real_index], dtype=torch.long).reshape(self.seq_len, -1)

        # logger.debug("\nReturn_data:", return_data.shape, "\n")  # = torch.Size([10, 11])
        # Ritorno le labels solo nella fase di Test, le label contengono i valori originali dell'amount
        if self.return_labels:
            return_labels = torch.tensor(self.labels[real_index], dtype=torch.double) #dtype=torch.long
            out_dict = {'input_ids': return_data, 'labels': return_labels}
            return out_dict
        return return_data

    def resample_train(self, train_indices, test_indices, eval_indices=[]):

        train_real_indices = [self.indices[i] for i in train_indices]
        test_real_indices = [self.indices[i] for i in test_indices]
        eval_real_indices = [self.indices[i] for i in eval_indices]
        train_labels = [self.labels[i] for i in train_real_indices]
        test_labels = [self.labels[i] for i in test_real_indices]
        eval_labels = [self.labels[i] for i in eval_real_indices]

        assert len(train_real_indices)+len(test_real_indices)+len(eval_real_indices) == len(self.indices)          
        assert len(train_labels)+len(test_labels)+len(eval_labels) == len(self.indices)          
        logger.info('Upsample training fraudulent samples.')
        train_real_indices = np.array(train_real_indices)
        train_labels = np.array(train_labels)
        logger.info(f'train labels shape: {train_labels.shape}')
        logger.info(f'train real indices shape: {train_real_indices.shape}')
        non_fraud_real_indices = train_real_indices[np.all(train_labels==0, axis=1)]
        non_fraud_labels = train_labels[np.all(train_labels==0, axis=1)]
        logger.info(f'non fraud indices shape: {non_fraud_real_indices.shape}')
        logger.info(f'non fraud labels shape: {non_fraud_labels.shape}')
        fraud_real_indices = train_real_indices[np.any(train_labels, axis=1)]
        fraud_labels = train_labels[np.any(train_labels, axis=1)]
        logger.info(f'fraud indices shape: {fraud_real_indices.shape}')
        logger.info(f'fraud labels shape: {fraud_labels.shape}')

        if fraud_labels.shape[0] < non_fraud_labels.shape[0]:
            logger.info(f'UPSAMPLING FRAUDS')
            fraud_upsample_real_indices = resample(fraud_real_indices, replace=True, n_samples=non_fraud_labels.shape[0], random_state=2022)
            logger.info(f'fraud upsample indices shape: {fraud_upsample_real_indices.shape}')
            train_real_indices = np.concatenate((fraud_upsample_real_indices,non_fraud_real_indices))
        else:
            logger.info(f'UPSAMPLING NON FRAUDS')
            non_fraud_upsample_real_indices = resample(non_fraud_real_indices, replace=True, n_samples=fraud_labels.shape[0], random_state=2022)
            logger.info(f'non fraud upsample indices shape: {non_fraud_upsample_real_indices.shape}')
            train_real_indices = np.concatenate((non_fraud_upsample_real_indices,fraud_real_indices))
        logger.info(f'new train indices shape: {train_real_indices.shape}')

        self.indices = list(train_real_indices)+eval_real_indices+test_real_indices
        train_indices = list(np.arange(train_real_indices.shape[0]))
        eval_indices = list(np.arange(train_real_indices.shape[0],train_real_indices.shape[0]+len(eval_real_indices)))
        test_indices = list(np.arange(train_real_indices.shape[0]+len(eval_real_indices),train_real_indices.shape[0]+len(eval_real_indices)+len(test_real_indices)))
        shuffle(train_indices)
        shuffle(eval_indices)
        shuffle(test_indices)
        logger.info(f'labels shape: {np.array(self.labels).shape}')
        logger.info(f'data shape: {np.array(self.data).shape}')
        logger.info(f'indices shape: {np.array(self.indices).shape}')
        assert len(self.data) == len(self.labels), f'data {len(self.data)} != labels {len(self.labels)}'
        assert len(self.indices) > len(self.labels), f'indices {len(self.indices)} <= labels {len(self.labels)}'
        return train_indices, test_indices, eval_indices

    def __len__(self):
        return len(self.indices)

    def save_vocab(self, vocab_dir):
        file_name = path.join(vocab_dir, f'input_vocab.nb')
        logger.info(f"saving input vocab at {file_name}")
        self.input_vocab.save_vocab(file_name)
        file_name = path.join(vocab_dir, f'output_vocab.nb')
        logger.info(f"saving output vocab at {file_name}")
        self.output_vocab.save_vocab(file_name)

    @staticmethod
    def label_fit_transform(column, enc_type="label"):
        if enc_type == "label":
            mfit = LabelEncoder()
        else:
            mfit = MinMaxScaler()
        mfit.fit(column)

        return mfit, mfit.transform(column)

    @staticmethod
    def timeEncoder(X):
        d = pd.to_datetime(dict(year=X['year'], month=X['month'], day=X['day']))

        # d = d.values.astype(int)  # originale
        # d = d.values.astype(np.int32)  # funziona 1
        d = d.apply(lambda x: x.value)  # funziona 2
        # int(datetime.datetime.utcnow().timestamp())
        return pd.DataFrame(d)

    @staticmethod
    def hourEncoder(X):
        hour = pd.to_datetime(X, format='%H:%M').dt.hour
        return hour

    @staticmethod
    def amountEncoder(X):
        amt = X
        amt = amt.apply(math.log)
        return pd.DataFrame(amt)

    @staticmethod
    def fraudEncoder(X):
        fraud = (X == 'Yes').astype(int)
        return pd.DataFrame(fraud)

    @staticmethod
    def nanNone(X):
        return X.where(pd.notnull(X), 'None')

    @staticmethod
    def nanZero(X):
        return X.where(pd.notnull(X), 0)

    def combinatorial_quantization(self, data):
        figures_list = [None]*self.num_figures
        data = [int(d*(self.base**self.comma)) for d in data]
        for i in range(self.num_figures):
            # logger.debug(f'computing figures {i}')
            # logger.debug(f'data is currently {data[:15]}')
            figures_data = np.array([int(d%self.base) for d in data])
            data = (data-figures_data)/self.base
            # logger.debug(f'figures computed are {figures_data[:15]}')
            # logger.debug(f'data has become {data[:15]}')
            figures_list[i] = figures_data
        return figures_list

    def _quantization_binning(self, data, numerical=False):
        num_bins = self.numerical_num_bins if numerical else self.num_bins
        qtls = np.arange(0.0, 1.0 + 1 / num_bins, 1 / num_bins)
        bin_edges = np.quantile(data, qtls, axis=0)
        # logger.debug(f'bin edges : {bin_edges}')
        bin_edges = np.sort(np.unique(bin_edges))
        # logger.debug(f'sorted bin edges : {bin_edges}')
        bin_widths = np.diff(bin_edges, axis=0)
        bin_centers = bin_edges[:-1] + bin_widths / 2
        return bin_edges, bin_centers, bin_widths

    def _quantize(self, inputs, bin_edges, numerical=False):
        num_bins = self.numerical_num_bins if numerical else self.num_bins
        quant_inputs = np.zeros(inputs.shape[0])
        for i, x in enumerate(inputs):
            quant_inputs[i] = np.digitize(x, bin_edges)
        quant_inputs = quant_inputs.clip(1, num_bins) - 1  # Clip edges
        # quant_inputs = quant_inputs - 1
        return quant_inputs

    def user_level_data(self):
        assert not self.flatten

        self.data_mapping_by_col = {}
        for col in self.trans_table.columns:
            if col in self.input_vocab.token2id.keys():
                data_col_mapping = {tok: int(self.input_vocab.get_id(tok, col)) for tok in self.input_vocab.get_field_tokens(col)}
                self.data_mapping_by_col[col] = data_col_mapping
        
        for user_code, user_data in tqdm.tqdm(self.trans_table.groupby(by="customer_id", sort=False)):
            user_data = user_data.iloc[:self.seq_len]
            user_labels = user_data['label']
            user_data = user_data.drop(columns=['label', 'customer_id'])

            for col in user_data.columns:
                if col not in self.regression_fields:
                    user_data[col] = user_data[col].map(self.data_mapping_by_col[col])
            
            user_data = np.array(user_data)
            user_labels = np.array(user_labels)

            user_data_padded = np.full((self.seq_len, user_data.shape[1]), self.tokenizer_pad_token_id, dtype=user_data.dtype)
            user_data_padded[:user_data.shape[0], :] = user_data

            user_labels_padded = np.full((self.seq_len,), user_labels[0], dtype=user_labels.dtype)
            user_labels_padded[:user_labels.shape[0]] = user_labels

            self.data.append(user_data_padded)
            self.labels.append(user_labels_padded)


    def format_trans(self, trans_lst, column_names):
        # change: -2 se isFraud è fra i campi, -1 se non c'è
        trans_lst = list(divide_chunks(trans_lst, len(self.input_vocab.field_keys) - 2))  # 2 to ignore isFraud and SPECIAL
        user_vocab_ids = []
        for trans in trans_lst:
            vocab_ids = []

            for jdx, field in enumerate(trans):
                if column_names[jdx] in self.regression_fields:
                    vocab_id = field
                else:
                    vocab_id = self.input_vocab.get_id(field, column_names[jdx])
                vocab_ids.append(vocab_id)

            if self.mlm and self.flatten and (not self.baseline):  # only add [SEP] for BERT + flatten scenario
                sep_id = self.input_vocab.get_id(self.input_vocab.sep_token, special_token=True)
                vocab_ids.append(sep_id)
            user_vocab_ids.append(vocab_ids)

        return user_vocab_ids


    def prepare_samples(self):
        logger.info("preparing user level data...")
        self.user_level_data()

        self.indices = list(range(len(self.labels)))
        assert len(self.data) == len(self.labels) == len(self.indices)

        # ncols = total fields - 1 (special tokens) - 1 fraud
        self.ncols = len(self.input_vocab.field_keys) - 2 + (1 if (self.mlm and self.flatten and (not self.baseline)) else 0)
        logger.info(f"ncols: {self.ncols}")
        logger.info(f"no of samples {len(self.data)}")


    def get_csv(self, fname):
        data = pd.read_csv(fname, nrows=self.nrows)
        if self.user_ids:
            logger.info(f'Filtering data by user ids list: {self.user_ids}...')
            self.user_ids = map(int, self.user_ids)
            data = data[data['customer_id'].isin(self.user_ids)]

        self.nrows = data.shape[0]
        logger.info(f"read data : {data.shape}")
        return data

    def write_csv(self, data, fname):
        logger.info(f"writing to file {fname}")
        data.to_csv(fname, index=False)

    def init_vocab(self):
        cols = list(self.trans_table.columns)
        if self.skip_user:
            cols.remove("customer_id")

        self.input_vocab.set_field_keys(cols)
        self.output_vocab.set_field_keys(cols)

        for column in cols:
            if column in self.regression_fields:
                if not self.quantize_regression_output:
                    val = 1
                    self.output_vocab.set_id(val, column)
                else:
                    for val in sorted(self.quantized_regression_values[column]):
                        self.output_vocab.set_id(val, column)
            else:
                unique_values = self.trans_table[column].value_counts(sort=True).to_dict()  # returns sorted
                for val in unique_values:
                    self.input_vocab.set_id(val, column)
                    self.output_vocab.set_id(val, column)

        logger.info(f"columns used for vocab: {list(cols)}")
        logger.info(f"total input vocabulary size: {len(self.input_vocab.id2token)}")
        logger.info(f"total output vocabulary size: {len(self.output_vocab.id2token)}")

        for column in cols:
            vocab_size = len(self.input_vocab.token2id[column])
            logger.info(f"column : {column}, input vocab size : {vocab_size}")
            vocab_size = len(self.output_vocab.token2id[column])
            logger.info(f"column : {column}, output vocab size : {vocab_size}")

    def encode_data(self):
        dirname = path.join(self.root, "preprocessed")
        fname = f'{self.fname}.encoded.csv'

        data_file = path.join(self.root, self.fname)
        data = pd.read_parquet(data_file)
        # data = data[self.configs["columns_to_select"]]

        # data = data.head(10000)

        parent_data = pd.read_parquet(path.join(self.root, "customers.parquet"))
        parent_data['label'] = (parent_data['age'].astype(float) > 30).astype(int)
        parent_data = parent_data[['customer_id', 'label']]

        if self.user_ids:
            logger.info(f'Filtering data by user ids list: {self.user_ids}...')
            self.user_ids = map(int, self.user_ids)
            data = data[data['customer_id'].isin(self.user_ids)]
        self.nrows = data.shape[0]
        logger.info(f"read data : {data.shape}")
        logger.info(f"{data_file} is read.")

        data = data.merge(parent_data, on="customer_id")

        # At the moment we are sorting by User, Card, Year, Month, Day, Time; considering card as User
        # TODO: Sort records by User and sort the records of the same user by datetime
        # data = data.sort_values(['customer_id', 'Year', 'Month', 'Day', 'Time'])

        # Index(['customer_id', 'date', 'year', 'month', 'day', 'transaction_amt', 'merchant_mcc', 'merchant_id'],

        # Amount Encoder
        logger.info("Amount Encoder.")
        data['transaction_amt'] = data['transaction_amt'].astype(float)
        # logger.debug("\n Dataframe completo:\n", data, "\n")  # data.to_csv("prova_prova.csv", index=False)

        data['transaction_amt'] = data['transaction_amt'].apply(lambda val_amt: max(1, val_amt))
        if 'transaction_amt' not in self.comb_fields:
            data['transaction_amt'] = data['transaction_amt'].apply(math.log)

        logger.info("timestamp fit transform")
        timestamp = self.timeEncoder(data[['year', 'month', 'day']])
        timestamp_fit, timestamp = self.label_fit_transform(timestamp, enc_type="time")
        self.encoder_fit['Timestamp'] = timestamp_fit
        data['Timestamp'] = timestamp

        logger.info("timestamp quant transform")
        col_data = np.array(data['Timestamp'])
        bin_edges, bin_centers, bin_widths = self._quantization_binning(col_data)
        data['Timestamp'] = self._quantize(col_data, bin_edges)
        self.encoder_fit["Timestamp-Quant"] = [bin_edges, bin_centers, bin_widths]

        sub_columns = ['merchant_mcc', 'year', 'month', 'day']

        logger.info("label-fit-transform.")
        for col_name in tqdm.tqdm(sub_columns):
            col_data = data[col_name]
            col_fit, col_data = self.label_fit_transform(col_data)
            self.encoder_fit[col_name] = col_fit
            data[col_name] = col_data

        col_name = 'transaction_amt'
        col_data = np.array(data['transaction_amt'])
        is_numerical_amount = ('transaction_amt' in self.numerical_fields)
        if 'transaction_amt' in self.regression_fields:
            if self.normalize_regression:
                col_data = col_data.reshape(-1, 1)
                amount_fit, col_data = self.label_fit_transform(col_data, enc_type="transaction_amt")
                self.encoder_fit['transaction_amt'] = amount_fit
            bin_edges, bin_centers, bin_widths = self._quantization_binning(col_data, numerical=is_numerical_amount)
            quantized_data = self._quantize(col_data, bin_edges, numerical=is_numerical_amount)
            self.encoder_fit["transaction_amt-Quant"] = [bin_edges, bin_centers, bin_widths]
            self.quantized_regression_edges[col_name] = bin_edges
            self.quantized_regression_values[col_name] = list(np.unique(quantized_data))
            logger.debug(f'quantized regression edges {bin_edges}')
            logger.debug(f'quantized regression values {self.quantized_regression_values[col_name]}')
            data['transaction_amt'] = col_data
        elif 'transaction_amt' in self.comb_fields:
            figures_list = self.combinatorial_quantization(col_data)
            for i, figure_array in enumerate(figures_list):
                col_name = f'transaction_amt_figure_{i}'
                data[col_name] = figure_array
        else:  # quantization
            logger.info("amount quant transform")
            bin_edges, bin_centers, bin_widths = self._quantization_binning(col_data, numerical=is_numerical_amount)
            data['transaction_amt'] = self._quantize(col_data, bin_edges, numerical=is_numerical_amount)
            self.encoder_fit["transaction_amt-Quant"] = [bin_edges, bin_centers, bin_widths]

        columns_to_select = self.columns_to_select.copy()
        if 'customer_id' not in columns_to_select:
            columns_to_select.insert(0, 'customer_id')
        columns_to_select.append('label')
        if 'transaction_amt' in self.comb_fields:
            start_idx = columns_to_select.index('transaction_amt')
            for i in range(self.num_figures):
                col_name = f'transaction_amt_figure_{self.num_figures - i - 1}'
                columns_to_select.insert(start_idx+i, col_name)
            columns_to_select.remove('transaction_amt')
        self.trans_table = data[columns_to_select]

        logger.info(f"writing cached csv to {path.join(dirname, fname)}")
        if not path.exists(dirname):
            os.mkdir(dirname)
        self.write_csv(self.trans_table, path.join(dirname, fname))

        encoder_fname = path.join(dirname, f'{self.fname}.encoder_fit.pkl')
        logger.info(f"writing cached encoder fit to {encoder_fname}")
        pickle.dump(self.encoder_fit, open(encoder_fname, "wb"))


def funct_segno(x):
    if x >= 0:
        return "positive"
    else:
        return "negative"


class TransactionDatasetEmbedded(TransactionDataset):
    def __init__(self,
                pretrained_model,              
                raw_dataset):

        self.pretrained_model = pretrained_model.to(device)
        self.raw_dataset = raw_dataset
        self.input_vocab = raw_dataset.input_vocab
        self.output_vocab = raw_dataset.output_vocab
        assert self.raw_dataset.return_labels
        self.data = []
        self.labels = []
        self.indices = []
        self.extract_embeddings()

    @torch.inference_mode()
    def extract_embeddings(self, batch_size=20):

        for start in tqdm.tqdm(range(0,len(self.raw_dataset)-batch_size, batch_size)):
            batch = torch.tensor([])
            for index in range(start,start+batch_size):
                inputs = self.raw_dataset.__getitem__(index)
                transaction = inputs['input_ids'].unsqueeze(0).to(device)
                self.labels.append(inputs['labels'])
                if batch.shape[0]>0:
                    batch = torch.cat((batch, transaction), 0)
                else:
                    batch = transaction.clone()
            outputs = self.pretrained_model(batch)
            # assert len(outputs)==2
            outputs = outputs[-1]
            embeddings = [outputs[i].squeeze().cpu() for i in range(outputs.shape[0])]
            self.data.extend(embeddings)

        self.indices = list(range(len(self.labels)))
        assert len(self.data)==len(self.labels)==len(self.indices)

    def resample_train(self, train_indices, test_indices, eval_indices=[]):

        train_real_indices = [self.indices[i] for i in train_indices]
        test_real_indices = [self.indices[i] for i in test_indices]
        eval_real_indices = [self.indices[i] for i in eval_indices]
        train_labels = [self.labels[i].numpy() for i in train_real_indices]
        test_labels = [self.labels[i] for i in test_real_indices]
        eval_labels = [self.labels[i] for i in eval_real_indices]

        assert len(train_real_indices)+len(test_real_indices)+len(eval_real_indices) == len(self.indices)          
        assert len(train_labels)+len(test_labels)+len(eval_labels) == len(self.labels)            
        logger.info('Upsample training fraudulent samples.')
        train_real_indices = np.array(train_real_indices)
        train_labels = np.array(train_labels)
        logger.info(f'train labels shape: {train_labels.shape}')
        logger.info(f'train real indices shape: {train_real_indices.shape}')
        non_fraud_real_indices = train_real_indices[np.all(train_labels==0, axis=1)]
        non_fraud_labels = train_labels[np.all(train_labels==0, axis=1)]
        logger.info(f'non fraud indices shape: {non_fraud_real_indices.shape}')
        logger.info(f'non fraud labels shape: {non_fraud_labels.shape}')
        fraud_real_indices = train_real_indices[np.any(train_labels, axis=1)]
        fraud_labels = train_labels[np.any(train_labels, axis=1)]
        logger.info(f'fraud indices shape: {fraud_real_indices.shape}')
        logger.info(f'fraud labels shape: {fraud_labels.shape}')

        if fraud_labels.shape[0] < non_fraud_labels.shape[0]:
            logger.info(f'UPSAMPLING FRAUDS')
            fraud_upsample_real_indices = resample(fraud_real_indices, replace=True, n_samples=non_fraud_labels.shape[0], random_state=2022)
            logger.info(f'fraud upsample indices shape: {fraud_upsample_real_indices.shape}')
            train_real_indices = np.concatenate((fraud_upsample_real_indices,non_fraud_real_indices))
        else:
            logger.info(f'UPSAMPLING NON FRAUDS')
            non_fraud_upsample_real_indices = resample(non_fraud_real_indices, replace=True, n_samples=fraud_labels.shape[0], random_state=2022)
            logger.info(f'non fraud upsample indices shape: {non_fraud_upsample_real_indices.shape}')
            train_real_indices = np.concatenate((non_fraud_upsample_real_indices,fraud_real_indices))
        logger.info(f'new train indices shape: {train_real_indices.shape}')

        self.indices = list(train_real_indices)+eval_real_indices+test_real_indices
        train_indices = list(np.arange(train_real_indices.shape[0]))
        eval_indices = list(np.arange(train_real_indices.shape[0],train_real_indices.shape[0]+len(eval_real_indices)))
        test_indices = list(np.arange(train_real_indices.shape[0]+len(eval_real_indices),train_real_indices.shape[0]+len(eval_real_indices)+len(test_real_indices)))
        shuffle(train_indices)
        shuffle(eval_indices)
        shuffle(test_indices)
        logger.info(f'labels shape: {np.array(self.labels).shape}')
        logger.info(f'data shape: {np.array(self.data).shape}')
        logger.info(f'indices shape: {np.array(self.indices).shape}')
        assert len(self.data) == len(self.labels), f'data {len(self.data)} != labels {len(self.labels)}'
        assert len(self.indices) > len(self.labels), f'indices {len(self.indices)} <= labels {len(self.labels)}'
        return train_indices, test_indices, eval_indices

    def __getitem__(self, index):

        real_index = self.indices[index]
        out_dict = {'input_ids': self.data[real_index], 'labels': self.labels[real_index]}
        return out_dict
