from loguru import logger

CONFIG_DICT = {

    ### argomenti temporanei per togliere args.py
    'seed': 9,
    'nrows': None,
    'cached': False,
    'user_ids': None,

    ### BASE PATHS CONFIGS
    'data_root': './data/pollution',
    'output_dir': './risultati_pretraining_prsa',

    ### DATASET CONFIGS
    'seq_len': 10,
    'stride': 10,
    'percentage_pretraining': 1.0,
    'columns_to_select': ['station',
                          'timestamp',
                        #    'year', 'month', 'day', 'hour',
                          'SO2',
                          'NO2',
                          'CO',
                          'O3',
                          'TEMP',
                          'PRES',
                          'DEWP',
                          'RAIN',
                          'wd',
                          'WSPM'],
    'skip_station': False,
    # parameters for timestamp
    'time_col_names': ['year', 'month', 'day', 'hour'],
    'use_splitted_timestamp': False,
    'consider_numerical_timestamp': False,
    # parameters for regression
    'regression_fields': ['SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','WSPM'],
    'normalize_regression': True,
    'regression_input_type': 'linear',
    'd_pos_enc': 8,
    'quantize_regression_output': False,
    # parameters for combinatorial quantization
    'combinatorial_quantization_fields': ['SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','WSPM'],
    'num_figures': 4,
    'combinatorial_quantization_base': 10,
    'combinatorial_quantization_comma': 0,
    # parameters for standard quantization
    'numerical_fields': ['SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','WSPM'],
    'numerical_num_bins': 50,
    'quantize_num_bins': 50,
    # parameters for changing how the initial embedding is constructed
    'embedding_type': 'no',
    'field_emb_size': 0,

    ### EXPERIMENT OUTPUT CONFIG
    # _ flat/hierarc _ (ce) _ hidden size _ (field hidden size) _ seq len _ (client split)
    'experiment_name': 'test_prsa',
    'state_dict_path': '',

    ### MODEL CONFIGS
    'flatten': False,
    'field_ce': True,
    'regression': False,
    'combinatorial_quantization': False,
    'model_head': 'linear',  ## Can be: 'linear', 'encoder', 'decoder'
    'output_hidden_states': True,

    # num_attention_heads=self.ncols*2
    # 'field_hidden_size': 110,
    # 'hidden_size': 770,
    # num_attention_heads=self.ncols
    'field_hidden_size': 72,
    'hidden_size': 792,

    ### TRAINING AND TEST CONFIGS
    'batch_size': 30,
    'test_batch_size': 30,
    'epochs': 100,
    'val_split_percentage': 0.0,
    'test_split_percentage': 0.0,
    "categorical_label_smoothing": False,
    "numerical_label_smoothing": False,
    'loss_regression': 'L2',  ### possible regression losses are L1, L2, L1smooth
    'loss_regression_lambda': 50,

    ###CONFIG_BERT masking methos
    # some configs needed only for bert model
    'mlm_prob': 0.15,
    'mask_line': False,
    'mask_line_probability': 0.05,
    'mask_specific_dict': False,
    'mask_all_timestamp': False,
}

CONFIG_FINETUNING = {
    #configs for the models that have to be finetuned on fraud detection task
    'use_embeddings': False,
    'create_equal_parameters_baselines': True,
    'normalized_target': True,
    'pretrained_model_folder': 'test_prsa',
    'pretrained_model_checkpoint': None,
    'load_weights': True,
    'output_dir': './risultati_finetuning',
    'experiment_name': 'test_prsa_finetuning',

    'use_minimal_feature': True,
    'aggregate_on_time': True,
    'use_catboost': False,
    'use_categorical': False,
    'use_onehot': False,

    'model_type': 'lstm',
    
    'hidden_size': 792,
    'num_layers_lstm': 1,
    'num_layers_mlp': 4,

    'seq_len': 10,
    'stride': 10,
    'batch_size': 20,
    'epochs': 100,

    ### parameters for dataset creation when no pretrained model is used
    'val_split_percentage': 0.2,
    'test_split_percentage': 0.2,
    'nrows': None,
    'cached': True,
    'user_ids': None,
    'segno_amount': 0,  # 0 non usa segno, 1 usa campo aggiuntivo, 2 somma il valore assoluto del min
    'skip_station': False,
    'columns_to_select': ['station',
                          'timestamp',
                          #'year', 'month', 'day', 'hour',
                          'SO2',
                          'NO2',
                          'CO',
                          'O3',
                          'TEMP',
                          'PRES',
                          'DEWP',
                          'RAIN',
                          'wd',
                          'WSPM'],
    ###### DO NOT CHANGE ARE ONLY FOR BASELINE ######
    'flatten': True,
    'regression': False,
    'quantize_regression_output': False,
    'combinatorial_quantization': False,
    'quantize_num_bins':10,
    'numerical_fields':[],
    'regression_fields':[],
    'combinatorial_quantization_fields':[],
    'numerical_num_bins': None,
    'normalize_regression': True,
    'num_figures': None,
    'combinatorial_quantization_base': None,
    'combinatorial_quantization_comma': None,
    ###### DO NOT CHANGE ARE ONLY FOR BASELINE ######

}

def check_configs():

    if CONFIG_DICT['skip_station'] and 'station' in CONFIG_DICT['columns_to_select']:
        CONFIG_DICT['columns_to_select'].remove("station")

    ## check that combinatorial quantization parameters are coherent
    if not CONFIG_DICT['combinatorial_quantization']:
        CONFIG_DICT['combinatorial_quantization_fields'] = []
    else:
        CONFIG_DICT['numerical_fields'] = []
        
    ## check that regression parameters are coherent
    if not CONFIG_DICT['regression']:
        CONFIG_DICT['regression_fields'] = []
    else:
        CONFIG_DICT['numerical_fields'] = []
    if not CONFIG_DICT['regression']:
        CONFIG_DICT['normalize_regression'] = False

    if CONFIG_DICT['numerical_label_smoothing']:
        CONFIG_DICT['loss_numerical'] = 'cross_entropy_smooth_local'
    else:
        CONFIG_DICT['loss_numerical'] = 'cross_entropy'
    if CONFIG_DICT['categorical_label_smoothing']:
        CONFIG_DICT['loss_categorical'] = 'cross_entropy_smooth'
    else:
        CONFIG_DICT['loss_categorical'] = 'cross_entropy'

    assert len(CONFIG_FINETUNING['numerical_fields'])==0
    assert len(CONFIG_FINETUNING['regression_fields'])==0
    assert len(CONFIG_FINETUNING['combinatorial_quantization_fields'])==0

    # if CONFIG_FINETUNING['create_equal_parameters_baselines']:
    #     assert CONFIG_DICT['field_hidden_size'] == CONFIG_FINETUNING['hidden_size']

    if CONFIG_DICT['use_splitted_timestamp']:
        col_to_select = set(CONFIG_DICT['columns_to_select'])
        col_to_select.discard('timestamp')
        col_to_select.update(['year', 'month', 'day', 'hour'])
        CONFIG_DICT['columns_to_select'] = list(col_to_select)
        assert 'timestamp' not in CONFIG_DICT['columns_to_select']
        assert 'year' in CONFIG_DICT['columns_to_select'] and 'month' in CONFIG_DICT['columns_to_select'] 
        assert 'day' in CONFIG_DICT['columns_to_select'] and 'hour' in CONFIG_DICT['columns_to_select']
    else:
        col_to_select = set(CONFIG_DICT['columns_to_select'])
        col_to_select.add('timestamp')
        for el in ['year', 'month', 'day', 'hour']:
            col_to_select.discard(el)
        CONFIG_DICT['columns_to_select'] = list(col_to_select)
        assert 'timestamp' in CONFIG_DICT['columns_to_select']
        assert 'year' not in CONFIG_DICT['columns_to_select'] and 'month' not in CONFIG_DICT['columns_to_select'] 
        assert 'day' not in CONFIG_DICT['columns_to_select'] and 'hour' not in CONFIG_DICT['columns_to_select']

    if CONFIG_DICT['embedding_type'] == 'field_embedding':
        assert CONFIG_DICT['field_emb_size']>0
    else:
        CONFIG_DICT['field_emb_size'] = 0

    num_cols = len(CONFIG_DICT['columns_to_select'])
    if CONFIG_DICT['combinatorial_quantization']:
        for field in CONFIG_DICT['combinatorial_quantization_fields']:
            num_cols += CONFIG_DICT['num_figures']-1
    if not CONFIG_DICT['hidden_size'] == CONFIG_DICT['field_hidden_size']*num_cols:
        logger.debug(f"{CONFIG_DICT['hidden_size']} should be {num_cols}*{CONFIG_DICT['field_hidden_size']}, \ncolumns to select are {CONFIG_DICT['columns_to_select']}")
        logger.debug(f"augmenting hidden size to {num_cols*CONFIG_DICT['field_hidden_size']}")
        CONFIG_DICT['hidden_size'] = CONFIG_DICT['field_hidden_size']*num_cols
    CONFIG_DICT['num_cols']=num_cols

    if 'quantize_regression_output' not in CONFIG_DICT.keys():
        CONFIG_DICT['quantize_regression_output'] = False

    if not CONFIG_DICT['use_splitted_timestamp']:
        CONFIG_DICT['mask_all_timestamp'] = False
        CONFIG_DICT['consider_numerical_timestamp'] = False

    if CONFIG_DICT['consider_numerical_timestamp']:
        if CONFIG_DICT['combinatorial_quantization']:
            CONFIG_DICT['combinatorial_quantization_fields'] += ['year', 'month', 'day', 'hour']     
        elif CONFIG_DICT['regression']:
            CONFIG_DICT['regression_fields'] += ['year', 'month', 'day', 'hour']
        else:
            CONFIG_DICT['numerical_fields'] += ['year', 'month', 'day', 'hour']