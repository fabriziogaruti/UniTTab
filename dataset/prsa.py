import torch
from torch.utils.data import Dataset
import glob
import pandas as pd
import numpy as np
from dataset.vocab import Vocabulary
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import tqdm
from os import path
from loguru import logger
import json
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PRSADataset(Dataset):
    def __init__(self,
                 configs,
                 root=")andrej/data/prsa/",
                 vocab_dir="output",
                 masked_lm=True,
                 return_labels=False,
                 baseline=False,
                 partial_ds=False):

        ## folder parameters and type of return are the only ones not in configs
        self.configs = configs
        self.root = root
        self.vocab_dir = vocab_dir
        self.return_labels = return_labels
        self.mlm = masked_lm
        self.baseline = baseline
        self.partial_ds = partial_ds

        self.input_vocab = Vocabulary()
        self.output_vocab = Vocabulary()
        self.encoding_fn = {}
        self.quantized_regression_values = {}
        self.quantized_regression_edges = {}
        self.target_cols = ['PM2.5', 'PM10']

        logger.info(f'DATASET CONFIGURATION FILE: \n{json.dumps(configs, indent=4)}')
        # Valori commentati erano i default
        self.skip_station = self.configs['skip_station']
        self.flatten = self.configs['flatten']
        self.stride = self.configs['stride']
        self.seq_len = self.configs['seq_len']  # seq_len = 10
        self.columns_to_select = self.configs['columns_to_select']

        # parameters to decide how to handle continuous variables
        self.flatten = self.configs['flatten']
        self.regression = self.configs['regression']
        self.quantize_regression_output = self.configs['quantize_regression_output']
        # quantization parameters
        self.num_bins = self.configs['quantize_num_bins']
        # regression parameters
        self.regression_fields = self.configs['regression_fields']
        self.normalize_regression = self.configs['normalize_regression']
        # combinatorial quantization parameters
        self.comb_fields = self.configs['combinatorial_quantization_fields']
        self.num_figures = self.configs['num_figures']
        self.base = self.configs['combinatorial_quantization_base']
        self.comma = self.configs['combinatorial_quantization_comma']

        self.ncols = None
        self.samples, self.targets = [], []

        self.numerical_columns = ['SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
        self.other_columns = ['wd', 'station', 'PM2.5', 'PM10', 'year', 'month', 'day', 'hour']

        self.setup()

    def __getitem__(self, index):
        if self.flatten:
            if self.regression:
                return_data = torch.tensor(self.samples[index], dtype=torch.float)
            else:
                return_data = torch.tensor(self.samples[index], dtype=torch.long)
        else:
            if self.regression:
                return_data = torch.tensor(self.samples[index], dtype=torch.float).reshape(self.seq_len, -1)
            else:
                return_data = torch.tensor(self.samples[index], dtype=torch.long).reshape(self.seq_len, -1)

        if self.return_labels:
            target = torch.tensor(self.targets[index], dtype=torch.float)
            out_dict = {'input_ids': return_data, 'labels': target}
            return out_dict
        return return_data

    def __len__(self):
        return len(self.samples)

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

    def _quantization_binning(self, data):
        qtls = np.arange(0.0, 1.0 + 1 / self.num_bins, 1 / self.num_bins)
        bin_edges = np.quantile(data, qtls, axis=0)
        # logger.debug(f'bin edges : {bin_edges}')
        bin_edges = np.sort(np.unique(bin_edges))
        # logger.debug(f'sorted bin edges : {bin_edges}')
        bin_widths = np.diff(bin_edges, axis=0)
        bin_centers = bin_edges[:-1] + bin_widths / 2
        return bin_edges, bin_centers, bin_widths

    def _quantize(self, inputs, bin_edges):
        quant_inputs = np.zeros(inputs.shape[0])
        for i, x in enumerate(inputs):
            quant_inputs[i] = np.digitize(x, bin_edges)
        quant_inputs = quant_inputs.clip(1, self.num_bins) - 1
        return quant_inputs

    @staticmethod
    def label_fit_transform(column, enc_type="label"):
        if enc_type == "label":
            mfit = LabelEncoder()
        else:
            mfit = MinMaxScaler()
        mfit.fit(column)
        return mfit, mfit.transform(column)

    @staticmethod
    def time_fit_transform(column):
        mfit = MinMaxScaler()
        mfit.fit(column)
        return mfit, mfit.transform(column)

    @staticmethod
    def timeEncoder(X):
        d = pd.to_datetime(dict(year=X['year'], month=X['month'], day=X['day'], hour=X['hour']))
        d = d.apply(lambda x: x.value)
        return pd.DataFrame(d)

    def setup(self):
        data = self.read_data()
        '''
        year 	month 	day 	hour 	PM2.5 	PM10 	SO2 	NO2 	
        CO 	O3 	TEMP 	PRES 	DEWP 	RAIN 	wd 	WSPM 	station
        '''

        cols_for_bins = []
        cols_for_bins += ['timestamp']

        data_cols = ['year', 'month', 'day', 'hour']
        timestamp = self.timeEncoder(data[data_cols])
        timestamp_fit, timestamp = self.time_fit_transform(timestamp)
        self.encoding_fn['timestamp'] = timestamp_fit
        data['timestamp'] = timestamp

        cols_for_bins += self.numerical_columns
        columns_to_select = self.columns_to_select.copy()
        for col in cols_for_bins:
            col_data = np.array(data[col])
            if col in self.regression_fields:
                logger.debug(f'preprocessing regression field {col}')
                if self.normalize_regression:
                    col_data = col_data.reshape(-1, 1)
                    col_fit, col_data = self.label_fit_transform(col_data, enc_type="amount")
                    self.encoding_fn['Amount'] = col_fit
                bin_edges, bin_centers, bin_widths = self._quantization_binning(col_data)
                quantized_data = self._quantize(col_data, bin_edges)
                self.quantized_regression_edges[col] = bin_edges
                self.quantized_regression_values[col] = list(np.unique(quantized_data))
                logger.debug(f'quantized regression edges {bin_edges}')
                logger.debug(f'quantized regression values {self.quantized_regression_values[col]}')
                data[col] = col_data
            elif col in self.comb_fields:
                figures_list = self.combinatorial_quantization(col_data)
                for i, figure_array in enumerate(figures_list):
                    col_name = f'{col}_figure_{i}'
                    data[col_name] = figure_array
                    columns_to_select.append(col_name)
                    cols_for_bins.append(col_name)
                columns_to_select.remove(col)
            else: # quantization
                bin_edges, bin_centers, bin_widths = self._quantization_binning(col_data)
                data[col] = self._quantize(col_data, bin_edges)
                self.encoding_fn[col] = [bin_edges, bin_centers, bin_widths]

        final_cols = []
        for col in (cols_for_bins + self.other_columns):
            if (col in columns_to_select) or (col in self.target_cols):
                final_cols.append(col)
        self.final_cols = final_cols
        logger.info(f'final columns are : {final_cols}')

        for target_col in self.target_cols:
            target_col_df = pd.DataFrame(data[target_col])
            target_col_fit, target_col_df_trans = self.time_fit_transform(target_col_df)
            self.encoding_fn[target_col] = target_col_fit
            data[target_col] = target_col_df_trans
            # Usage: self.encoding_fn['PM2.5'].inverse_transform(X)

        logger.info(f'total available columns are {len(data.columns)} -> {data.columns}')
        self.data = data[final_cols]
        logger.info(f'selected columns are {len(self.data.columns)} -> {self.data.columns}')
        self.init_vocab()
        self.prepare_samples()
        self.save_vocab(self.vocab_dir)

    def prepare_samples(self):
        groups = self.data.groupby('station')
        for group in tqdm.tqdm(groups):
            station_name, station_data = group

            nrows = station_data.shape[0]
            nrows = nrows - self.seq_len

            logger.info(f"{station_name} : {nrows}")
            for sample_id in range(0, nrows, self.stride):
                sample, target = [], []
                for tid in range(0, self.seq_len):
                    row = station_data.iloc[sample_id + tid]
                    for col_name, col_value in row.items():
                        if self.skip_station:
                            if col_name == 'station':
                                continue
                        if col_name not in self.target_cols:
                            if col_name in self.regression_fields:
                                vocab_id = col_value
                            else:
                                vocab_id = self.input_vocab.get_id(col_value, col_name)
                            sample.append(vocab_id)

                    if self.mlm and self.flatten and (not self.baseline):
                        sep_id = self.input_vocab.get_id(self.input_vocab.sep_token, special_token=True)
                        sample.append(sep_id)
                    target.append(row[self.target_cols].tolist())

                self.samples.append(sample)
                self.targets.append(target)
            # break

        assert len(self.samples) == len(self.targets)
        logger.info(f"total samples {len(self.samples)}")

        logger.info(f'input vocab field keys are : {self.input_vocab.get_field_keys(ignore_special=True)}')
        logger.info(f'output vocab field keys are : {self.output_vocab.get_field_keys(ignore_special=True)}')

        self.ncols = len(self.input_vocab.get_field_keys(ignore_special=True))
        self.ncols += (1 if (self.mlm and self.flatten and (not self.baseline)) else 0)           
        
    def init_vocab(self):
        cols = list(self.data.columns)

        if self.skip_station:
            cols.remove('station')

        for col in self.target_cols:
            cols.remove(col)

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
                unique_values = self.data[column].value_counts(sort=True).to_dict()  # returns sorted
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

    def read_data(self):
        all_stations = None
        fnames = glob.glob(f"{self.root}/*.csv")
        for index, fname in enumerate(fnames):
            # Partial ds: e.g., limit number of stations to 3
            if self.partial_ds and index == 3:
                break
            station_data = pd.read_csv(fname)
            all_stations = pd.concat([all_stations, station_data], ignore_index=True)

        all_stations.drop(columns=['No'], inplace=True, axis=1)
        logger.info(f"shape (original)    : {all_stations.shape}")
        all_stations = all_stations.dropna()
        logger.info(f"shape (after nan removed): {all_stations.shape}")
        return all_stations

    def save_vocab(self, vocab_dir):
        file_name = path.join(vocab_dir, f'input_vocab.nb')
        logger.info(f"saving input vocab at {file_name}")
        self.input_vocab.save_vocab(file_name)
        file_name = path.join(vocab_dir, f'output_vocab.nb')
        logger.info(f"saving output vocab at {file_name}")
        self.output_vocab.save_vocab(file_name)

class PRSADatasetEmbedded(PRSADataset):
    def __init__(self,
                pretrained_model,              
                raw_dataset):

        self.pretrained_model = pretrained_model.to(device)
        self.raw_dataset = raw_dataset
        self.input_vocab = raw_dataset.input_vocab
        self.output_vocab = raw_dataset.output_vocab
        self.encoding_fn = raw_dataset.encoding_fn
        assert self.raw_dataset.return_labels
        self.samples = []
        self.targets = []
        self.extract_embeddings()

    @torch.inference_mode()
    def extract_embeddings(self, batch_size=20):

        logger.info(f'computing embedding with pretrained model')
        for start in tqdm.tqdm(range(0,len(self.raw_dataset)-batch_size, batch_size)):
            batch = torch.tensor([])
            for index in range(start,start+batch_size):
                inputs = self.raw_dataset.__getitem__(index)
                sequence = inputs['input_ids'].unsqueeze(0).to(device)
                self.targets.append(inputs['labels'])
                if batch.shape[0]>0:
                    batch = torch.cat((batch, sequence), 0)
                else:
                    batch = sequence.clone()
            outputs = self.pretrained_model(batch)
            outputs = outputs[-1]
            embeddings = [outputs[i].squeeze().cpu() for i in range(outputs.shape[0])]
            self.samples.extend(embeddings)

        assert len(self.samples)==len(self.targets)

    def __getitem__(self, index):

        out_dict = {'input_ids': self.samples[index], 'labels': self.targets[index]}
        return out_dict