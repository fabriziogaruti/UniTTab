import os
from os import path
import pandas as pd
import numpy as np
import json
import tqdm
import pickle
from loguru import logger
from sklearn.utils import resample
from random import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from dataset.vocab import EncoderVocab

import torch
from torch.utils.data.dataset import Dataset

from misc.utils import divide_chunks
from datetime import datetime

from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dataset.vocab import EncoderVocab

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_strdate(start, duration):
    date_time_obj = datetime.strptime(str(start), '%y%m%d')
    date_time_obj = date_time_obj + relativedelta(months=+duration)
    return int(datetime.strftime(date_time_obj, '%Y%m%d'))


class CzechDataset(Dataset):
    def __init__(self,
                 configs,
                 root="./data/card/",
                 fname="card_transaction_200",
                 vocab_dir="risultati_training",
                 return_labels=False,
                 masked_lm=True,
                 baseline=False, ):

        # folder parameters and type of return are the only ones not in configs
        self.configs = configs
        self.root = root
        self.fname = fname
        self.vocab_dir = vocab_dir
        self.return_labels = return_labels
        self.mlm = masked_lm
        self.baseline = baseline

        # initialize parameter for the encoder and the vocabulary to null
        self.encoder_vocab = EncoderVocab(
            encoder_fit=None,
            input_vocab=None,
            output_vocab=None
        )
        # load input and output vocabulary and the encoder fit
        self.get_encoder_vocab()

        self.input_vocab = self.encoder_vocab.input_vocab
        self.output_vocab = self.encoder_vocab.output_vocab
        # self.encoding_fn = {}  # TODO: rimosso
        self.quantized_regression_values = {}
        self.quantized_regression_edges = {}

        logger.info(f'DATASET CONFIGURATION FILE: \n{json.dumps(configs, indent=4)}')
        # parameters to reduce the dataset
        self.nrows = self.configs['nrows']

        self.seq_len = self.configs['seq_len']
        self.random_seq_start = self.configs['random_seq_start']
        self.columns_to_select = self.configs['columns_to_select']

        # parameters to decide how to handle continuous variables
        self.flatten = self.configs['flatten']
        self.regression = self.configs['regression']
        # self.quantize_regression_output = self.configs['quantize_regression_output']  # TODO: rimosso
        # quantization parameters
        self.num_bins = self.configs['quantize_num_bins']
        self.numerical_num_bins = self.configs['numerical_num_bins']
        self.numerical_fields = self.configs['numerical_fields']
        # regression parameters
        self.regression_fields = self.configs['regression_fields']
        self.normalize_regression = self.configs['normalize_regression']
        # combinatorial quantization parameters
        self.comb_fields = self.configs['combinatorial_quantization_fields']
        self.num_figures = self.configs['num_figures']
        self.base = self.configs['combinatorial_quantization_base']
        self.comma = self.configs['combinatorial_quantization_comma']

        # initialize some helper variables
        self.ncols = None  # pretrain_dataset.ncols
        self.encoder_fit = self.encoder_vocab.encoder_fit  # dizionario contenente nome colonna e lista dei margini di quantizzazione

        self.trans_table = None
        self.data, self.labels = [], []  # = self.samples, self.targets in prsa
        self.indices = []

        # preprocess data and create vocab
        self.encode_data()
        # self.init_vocab()
        self.prepare_samples()
        self.save_vocab()

    def __getitem__(self, index):
        real_index = self.indices[index]
        if self.flatten:
            if self.regression:
                return_data = torch.tensor(self.data[real_index], dtype=torch.float)
            else:
                return_data = torch.tensor(self.data[real_index], dtype=torch.long)
        else:
            if self.regression:
                return_data = torch.tensor(self.data[real_index], dtype=torch.float).reshape(-1, self.final_num_columns)
            else:
                return_data = torch.tensor(self.data[real_index], dtype=torch.long).reshape(-1, self.final_num_columns)

        if self.random_seq_start and return_data.shape[0]>self.seq_len:
            random_start = torch.randint(0,return_data.shape[0] - self.seq_len,())
            random_end = random_start+self.seq_len
            return_data = return_data[random_start:random_end,:]
        # logger.debug("\nReturn_data:", return_data.shape, "\n")  # = torch.Size([10, 11])
        # Ritorno le labels solo nella fase di Test, le label contengono i valori originali dell'amount
        if self.return_labels:
            return_labels = torch.tensor(self.labels[real_index], dtype=torch.double) #dtype=torch.long
            out_dict = {'input_ids': return_data, 'labels': return_labels}
            return out_dict
        return return_data

    def __len__(self):
        return len(self.indices)

    def save_vocab(self):
        file_name = path.join(self.vocab_dir, f'input_vocab.nb')
        logger.info(f"saving input vocab at {file_name}")
        self.input_vocab.save_vocab(file_name)
        file_name = path.join(self.vocab_dir, f'output_vocab.nb')
        logger.info(f"saving output vocab at {file_name}")
        self.output_vocab.save_vocab(file_name)

    def get_encoder_vocab(self):
        file_name = path.join(self.vocab_dir, f'encoder_vocab.pkl')
        with open(file_name, 'rb') as input_file:
            self.encoder_vocab = pickle.load(input_file)

    @staticmethod
    def label_fit_transform(column, enc_type="label"):
        if enc_type == "label":
            mfit = LabelEncoder()
        else:
            mfit = MinMaxScaler()
        mfit.fit(column)
        return mfit, mfit.transform(column)

    @staticmethod
    def dateTimeEncoder(X, table_name):
        if table_name == 'issued':
            x2 = X.apply(lambda x: (datetime.strptime(str(x), '%y%m%d %H:%M:%S')).timestamp() if pd.notnull(x) else -1)
        else:
            x2 = X.apply(lambda x: (datetime.strptime(str(x), '%y%m%d')).timestamp())
        return pd.DataFrame(x2)

    @staticmethod
    def loanStatusEncoder(X):
        loan = (X == 'A').astype(int)
        loan += ((X == 'B').astype(int) * 2)
        loan += (X == 'C').astype(int)
        loan += ((X == 'D').astype(int) * 2)
        loan = loan - 1
        return pd.DataFrame(loan)

    @staticmethod
    def nanNone(X):
        return X.where(pd.notnull(X), 'None')

    @staticmethod
    def convFloatStr(X):
        x2 = X.where(pd.notnull(X), 0).astype(int)
        x2 = x2.where(x2 != 0, 'None')
        return x2.astype(str)

    def combinatorial_quantization(self, data):
        figures_list = [None]*self.num_figures
        data = [int(d*(self.base**self.comma)) for d in data]
        for i in range(self.num_figures):
            # logger.debug(f'computing figures {i}')
            # logger.debug(f'data is currently {data[:15]}')
            figures_data = np.array([int(d % self.base) for d in data])
            data = (data-figures_data)/self.base
            # logger.debug(f'figures computed are {figures_data[:15]}')
            # logger.debug(f'data has become {data[:15]}')
            figures_list[i] = figures_data
        return figures_list

    def _quantization_binning(self, data):
        qtls = np.arange(0.0, 1.0 + 1 / self.num_bins, 1 / self.num_bins)
        bin_edges = np.quantile(data, qtls, axis=0)  # (num_bins + 1, num_features)
        bin_widths = np.diff(bin_edges, axis=0)
        bin_centers = bin_edges[:-1] + bin_widths / 2  # ()
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
        fname = path.join(self.root, f"preprocessed/{self.fname}.user.pkl")
        trans_data, trans_labels = [], []

        columns_names = list(self.trans_table.columns)
        columns_names.remove('account_id')
        if self.return_labels:
            for col_name in ['date_loan', 'amount_loan', 'duration', 'status', 'loan_id', ]:  # loan
                columns_names.remove(col_name)

        if self.return_labels:
            for user in tqdm.tqdm(self.trans_table['loan_id'].unique()):
                user_data = self.trans_table.loc[self.trans_table["loan_id"] == user]

                # user_trans = []
                # for idx, row in user_data.iterrows():
                #     row = list(row)
                #     # assumption that user is first field
                #     user_trans.extend(row[2:-1])  # todo: change: mettere :-1 se isFraud è fra i campi
                # # check on the last column if the estinction_date is not empty
                # user_label = int(str(row[-1]) != "")

                user_label2 = list(user_data.iloc[-1])[-5:]
                user_data2 = user_data.drop(user_data.columns[[0, -5, -4, -3, -2, -1]], axis=1)
                user_trans2 = list(user_data2.to_numpy().flatten())

                # assert user_label==user_label2, f'{user_label} should be {user_label2}'
                # assert user_trans==user_trans2, f'{user_trans} should be {user_trans2}'

                trans_data.append(user_trans2)
                trans_labels.append(user_label2)
        else:
            for user in tqdm.tqdm(self.trans_table['account_id'].unique()):
                user_data = self.trans_table.loc[self.trans_table["account_id"] == user]
                user_data2 = user_data.drop(user_data.columns[[0]], axis=1)
                user_trans2 = list(user_data2.to_numpy().flatten())
                trans_data.append(user_trans2)
                trans_labels.append([-1, -1, -1, -1, -1])

        with open(fname, 'wb') as cache_file:
            pickle.dump({"trans": trans_data, "labels": trans_labels, "columns": columns_names}, cache_file)

        # convert to str
        return trans_data, trans_labels, columns_names

    def format_trans(self, trans_lst, column_names):
        # change: -2 se isFraud è fra i campi, -1 se non c'è
        trans_lst = list(divide_chunks(trans_lst, len(self.input_vocab.field_keys) - 1))  # todo: 2 to ignore isFraud
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
        # Group by
        trans_data, trans_labels, columns_names = self.user_level_data()
        self.final_num_columns = len(columns_names)

        logger.info("creating transaction samples with vocab")
        for user_idx in tqdm.tqdm(range(len(trans_data))):
            user_row = trans_data[user_idx]
            user_labels = trans_labels[user_idx]

            # metto il valore di vocabolario
            user_row_ids = self.format_trans(user_row, columns_names)

            if self.random_seq_start:
                ids = user_row_ids[ : ]
            else:
            # filtro e prendo gli ultimi sample (100) (seq_len)
                ids = user_row_ids[max(0, len(user_row_ids) - self.seq_len) : ]
            ids_flattened = [idx for ids_lst in ids for idx in ids_lst]  # flattening
            if self.flatten and (not self.mlm) and (not self.baseline):  # for GPT2, need to add [BOS] and [EOS] tokens
                bos_token = self.input_vocab.get_id(self.input_vocab.bos_token, special_token=True)  # will be used for GPT2
                eos_token = self.input_vocab.get_id(self.input_vocab.eos_token, special_token=True)  # will be used for GPT2
                ids = [bos_token] + ids_flattened + [eos_token]

            self.data.append(ids_flattened)
            self.labels.append(user_labels[3])

        self.indices = list(range(len(self.labels)))
        assert len(self.data) == len(self.labels) == len(self.indices)

        # ncols = total fields - 1 (special tokens)
        self.ncols = len(self.input_vocab.field_keys) - 1 + (1 if (self.mlm and self.flatten and (not self.baseline)) else 0)
        logger.info(f"ncols: {self.ncols}")
        logger.info(f"no of samples {len(self.data)}")

    def get_csv(self, dataset_path):
        transaction = pd.read_csv(dataset_path / 'trans.asc', sep=';')
        account = pd.read_csv(dataset_path / 'account.asc', sep=';')
        loan = pd.read_csv(dataset_path / 'loan.asc', sep=';')
        district = pd.read_csv(dataset_path / 'district.asc', sep=';')
        disp = pd.read_csv(dataset_path / 'disp.asc', sep=';')
        card = pd.read_csv(dataset_path / 'card.asc', sep=';')

        # Rimuovere loan payments  # TODO: commenta
        transaction = transaction[transaction["k_symbol"] != 'UVER']

        # Rename Columns
        transaction = transaction.rename(columns={"date": "date_trans", "type": "type_trans", "amount": "amount_trans"})
        account = account.rename(columns={"date": "date_account", })
        card = card.rename(columns={"type": "type_card"})
        district = district.rename(columns={"A1": "district_id"})
        loan = loan.rename(columns={"amount": "amount_loan", "date": "date_loan", })

        print("Trans shape", transaction.shape)
        pd_join = transaction.merge(account, on='account_id', how='left')

        # Filter Disposition, type = owner
        disp = disp[disp['type'] == 'OWNER']
        disp_part = disp[["disp_id", "client_id", "account_id"]]

        pd_join = pd_join.merge(disp_part, on='account_id', how='left')
        pd_join = pd_join.merge(card, on='disp_id', how='left')
        pd_join = pd_join.merge(district, on='district_id', how='left')

        '''cols_to_select = [
            "date_trans", "type_trans", "operation", "amount_trans", "balance", "k_symbol", "bank", "account",  # transaction
            'account_id', "date_account", "frequency",  # account
            "type_card", "issued",  # card
            'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16',  # district
        ]'''

        if self.return_labels:
            print("Pd_join shape", pd_join.shape, "\n")
            print(pd_join.columns, "\n")

            loan['date_loan_end'] = loan.apply(lambda x: get_strdate(x.date_loan, x.duration), axis=1)
            pd_join = loan.merge(pd_join, on='account_id', how='left')
            print("Join with loan shape", pd_join.shape)  # date_x amount_x date_y amount_y

            '''# Da data di apertura alla data di chiusura
            sel_trans = pd_join[pd_join["date_trans"] > pd_join["date_loan"]]
            sel_trans = sel_trans[sel_trans["date_trans"] < sel_trans["date_loan_end"]]
            print("Shape con date filter:", sel_trans.shape, "\n")
            # print(sel_trans.columns)
            # print("\n")'''
            '''# Fino a data di chiusura  # TODO: decommenta
            pd_join = pd_join[pd_join["date_trans"] < pd_join["date_loan_end"]]
            print("Shape con date filter:", pd_join.shape, "\n")'''
            # Fino a data di apertura
            pd_join = pd_join[pd_join["date_trans"] < pd_join["date_loan"]]
            print("Shape con date filter:", pd_join.shape, "\n")

            '''cols_to_select.extend(['date_loan', 'amount_loan', 'duration', 'status', 'loan_id'])  # loan
            data = pd_join[cols_to_select]'''

        data = pd_join
        print("Data shape", data.shape, "\n")
        print(data.columns)

        if self.nrows != None:
            data = data.head(self.nrows)

        self.nrows = data.shape[0]
        logger.info(f"read data : {data.shape}")
        return data

    def write_csv(self, data, fname):
        logger.info(f"writing to file {fname}")
        data.to_csv(fname, index=False)

    def encode_data(self):
        dirname = path.join(self.root, "preprocessed")
        fname = f'{self.fname}.encoded.csv'
        # data_file = path.join(self.root, f"{self.fname}.csv")
        data = self.get_csv(Path(self.root))
        logger.info(f"{self.root} is read.")

        logger.info("nan resolution.")
        data['operation'] = self.nanNone(data['operation'])
        data['k_symbol'] = self.nanNone(data['k_symbol'])
        data['bank'] = self.nanNone(data['bank'])
        data['account'] = self.convFloatStr(data['account'])
        data['type_card'] = self.nanNone(data['type_card'])

        if self.return_labels:
            logger.info("target-encoder.")
            data['status'] = self.loanStatusEncoder(data['status'])

        # Create columns: ['Year', 'Month', 'Day']
        data['Year'], data['Month'], data['Day'] = zip(*data['date_trans'].apply(lambda x: [str(x)[:2], str(x)[2:4], str(x)[4:]]))

        logger.info("label-transform.")
        sub_columns = ['type_trans', 'operation', 'k_symbol', 'bank', 'account', 'frequency', 'type_card']
        for col_name in tqdm.tqdm(sub_columns):
            col_data = data[col_name]
            col_data = self.encoder_fit[col_name].transform(col_data)
            data[col_name] = col_data

        if self.return_labels:
            # logger.info("label-fit-transform.") # for loan
            col_name = 'duration'
            col_data = data[col_name]
            col_fit, col_data = self.label_fit_transform(col_data)
            self.encoder_fit[col_name] = col_fit
            data[col_name] = col_data

        time_columns = ['date_trans', 'date_account', 'issued']
        for col_name in time_columns:
            timestamp = self.dateTimeEncoder(data[col_name], col_name)
            timestamp[timestamp != -1] = self.encoder_fit[col_name].transform(timestamp[timestamp != -1])
            data[col_name] = timestamp
            logger.info(col_name + " timestamp quant transform")
            col_data = np.array(data[col_name])
            data[col_name] = self._quantize(col_data, self.encoder_fit[col_name + "-Quant"][0])

        if self.return_labels:
            # log.info("timestamp-fit-transform.") # for loan
            col_name = 'date_loan'
            logger.info(col_name + " timestamp fit transform")
            timestamp = self.dateTimeEncoder(data[col_name], col_name)
            timestamp_fit, timestamp[timestamp != -1] = self.label_fit_transform(timestamp[timestamp != -1], enc_type="time")
            self.encoder_fit[col_name] = timestamp_fit
            data[col_name] = timestamp
            logger.info(col_name + " timestamp quant transform")
            col_data = np.array(data[col_name])
            bin_edges, bin_centers, bin_widths = self._quantization_binning(col_data[col_data != -1])
            data[col_name] = self._quantize(col_data, bin_edges)
            self.encoder_fit[col_name + "-Quant"] = [bin_edges, bin_centers, bin_widths]

        # amount trans, balance
        # amount_columns = ['valore_vl','numero_num','saldo_stimato']
        amount_columns = ['amount_trans', 'balance']
        for col_name in amount_columns:
            col_data = np.array(data[col_name])
            is_numerical_amount = (col_name in self.numerical_fields)

            if col_name in self.regression_fields:
                if self.normalize_regression:
                    col_data = col_data.reshape(-1, 1)
                    col_data = self.encoder_fit[col_name].transform(col_data)

                quantized_data = self._quantize(col_data, self.encoder_fit[col_name + "-Quant"][0], numerical=is_numerical_amount)
                self.quantized_regression_edges[col_name] = self.encoder_fit[col_name + "-Quant"][0]
                self.quantized_regression_values[col_name] = list(np.unique(quantized_data))
                logger.debug(f'quantized regression edges {self.encoder_fit[col_name + "-Quant"][0]}')
                logger.debug(f'quantized regression values {self.quantized_regression_values[col_name]}')
                data[col_name] = col_data

            elif col_name in self.comb_fields:
                figures_list = self.combinatorial_quantization(col_data)
                for i, figure_array in enumerate(figures_list):
                    col_name = col_name+f'_figure_{i}'
                    data[col_name] = figure_array

            else:  # quantization
                logger.info(col_name+" amount quant transform")
                col_data = np.array(data[col_name])
                data[col_name] = self._quantize(col_data, self.encoder_fit[col_name + "-Quant"][0])

        # columns_to_select = ["hashed_ndg", "ident_prodotto_code", "transaction_date", "causale_code", "valore_vl",
        #                      "canale_code", "saldo_stimato"]
        columns_to_select = self.columns_to_select.copy()
        if 'account_id' in columns_to_select:
            columns_to_select.remove('account_id')
        columns_to_select.insert(0, 'account_id')

        if self.return_labels:
            # columns_to_select.append('status')
            columns_to_select.extend(['date_loan', 'amount_loan', 'duration', 'status', 'loan_id'])  # loan

        for col_name in amount_columns:
            if col_name in self.comb_fields:
                start_idx = columns_to_select.index(col_name)
                for i in range(self.num_figures):
                    col_name = 'Amount_'+col_name+f'_figure_{self.num_figures - i - 1}'
                    columns_to_select.insert(start_idx+i, col_name)
                columns_to_select.remove(col_name)

        self.trans_table = data[columns_to_select]

        logger.info(f"writing cached csv to {path.join(dirname, fname)}")
        if not path.exists(dirname):
            os.mkdir(dirname)
        self.write_csv(self.trans_table, path.join(dirname, fname))

        encoder_fname = path.join(dirname, f'{self.fname}.encoder_fit.pkl')
        logger.info(f"writing cached encoder fit to {encoder_fname}")
        pickle.dump(self.encoder_fit, open(encoder_fname, "wb"))


def collate_batch(examples, pad_token=0, seq_len=50):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""

    have_third_dim = examples[0].dim()==2
    if have_third_dim:
        third_dim = examples[0].size(1)
    # Check if padding is necessary.
    are_tensors_same_length = all(x.size(0) == seq_len for x in examples)
    if are_tensors_same_length:
        return torch.stack(examples, dim=0)

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    max_length = max(seq_len, max_length)
    # logger.debug(f'lengths in batch are {[x.size(0) for x in examples]}, padded to {max_length}')
    if have_third_dim:
        result = examples[0].new_full([len(examples), max_length, third_dim], pad_token)
    else:
        result = examples[0].new_full([len(examples), max_length], pad_token)
    for i, example in enumerate(examples):
        result[i, : example.shape[0]] = example
    return result


class CzechDatasetEmbedded(CzechDataset):
    def __init__(self,
                pretrained_model,
                raw_dataset):

        self.pretrained_model = pretrained_model.to(device)
        self.raw_dataset = raw_dataset
        self.input_vocab = raw_dataset.input_vocab
        self.pad_token = self.input_vocab.token2id[self.input_vocab.special_field_tag][self.input_vocab.pad_token][0]
        logger.debug(f'pad token is {self.pad_token}')
        self.output_vocab = raw_dataset.output_vocab
        self.ncols = raw_dataset.ncols
        self.seq_len = raw_dataset.seq_len
        assert self.raw_dataset.return_labels
        self.data = []
        self.labels = []
        self.indices = []
        self.extract_embeddings()

    @torch.inference_mode()
    def extract_embeddings(self, batch_size=20):

        logger.info(f'computing embedding with pretrained model')
        for start in tqdm.tqdm(range(0,len(self.raw_dataset), batch_size)):
            batch = torch.tensor([])
            end = min(start+batch_size, len(self.raw_dataset))
            items = [self.raw_dataset.__getitem__(index) for index in range(start,end)]
            inputs = [item['input_ids'] for item in items]
            labels = [item['labels'] for item in items]
            batch = collate_batch(inputs, pad_token=self.pad_token, seq_len=self.seq_len).to(device)
            # logger.debug(batch.shape)
            outputs = self.pretrained_model(batch)
            assert len(outputs)==2
            outputs = outputs[-1]
            embeddings = [outputs[i].squeeze().cpu() for i in range(outputs.shape[0])]
            self.data.extend(embeddings)
            self.labels.extend(labels)

        self.indices = list(range(len(self.labels)))
        assert len(self.data)==len(self.labels)==len(self.indices)
        assert len(self.data)==len(self.raw_dataset.data), f'computing embedding some sequences are lost : {len(self.data)}!={len(self.raw_dataset.data)}'


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
        fraud_upsample_real_indices = resample(fraud_real_indices, replace=True, n_samples=non_fraud_labels.shape[0], random_state=2022)
        logger.info(f'fraud upsample indices shape: {fraud_upsample_real_indices.shape}')
        train_real_indices = np.concatenate((fraud_upsample_real_indices,non_fraud_real_indices))
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
