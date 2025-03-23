from loguru import logger

CONFIG_DICT = {

    ### argomenti temporanei per togliere args.py
    'seed': 9,
    'nrows': None,
    'cached': False,
    'user_ids': None,

    ### BASE PATHS CONFIGS
    'data_root': './data/loan',
    'output_dir': './results_ablation_czech',
    'data_fname': 'trans_2',
    # 'data_fname': 'transactions_12months_100000',
    'data_fname_vocab': 'trans_2',

    ### DATASET CONFIGS
    'percentage_pretraining': 1.0,
    'random_seq_start': True,
    'seq_len': 100,
    'columns_to_select': ["date_trans", "type_trans", "operation", "amount_trans", "balance",
                             "k_symbol",  # transaction
                             'account_id', ], # "frequency",  # account
                             # "type_card", ],  # card
    # parameters for timestamp
    'time_col_names': ['Year', 'Month', 'Day'],
    'use_splitted_timestamp': False,
    # parameters for regression
    'regression_fields': ['amount_trans', 'balance'],
    'normalize_regression': True,
    'regression_input_type': 'linear',
    'd_pos_enc': 8,
    'quantize_regression_output': False,
    # parameters for combinatorial quantization
    'combinatorial_quantization_fields': ['amount_trans', 'balance'],
    'num_figures': 4,
    'combinatorial_quantization_base': 10,
    'combinatorial_quantization_comma': 0,
    # parameters for standard quantization
    'numerical_fields': ['amount_trans', 'balance'],
    'numerical_num_bins': 100,
    'quantize_num_bins': 10,
    # parameters for changing how the initial embedding is constructed
    'embedding_type': 'no',
    'field_emb_size': 0,

    ### EXPERIMENT OUTPUT CONFIG
    # _ flat/hierarc _ (ce) _ hidden size _ (field hidden size) _ seq len _ (client split)
    'experiment_name': 'test',
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
    'batch_size': 30, # 15,
    'test_batch_size': 30, # 15,
    'epochs': 20,
    'val_split_percentage': 0.0,
    'test_split_percentage': 0.0,
    "categorical_label_smoothing": False,
    "numerical_label_smoothing": False,
    'loss_regression': 'L1',  ### possible regression losses are L1, L2, L1smooth
    'loss_regression_lambda': 1,

    ###CONFIG_BERT masking methos
    # some configs needed only for bert model
    'mlm_prob': 0.15,
    'mask_line': False,
    'mask_line_probability': 0.05,
    'mask_specific_dict': False,
    'mask_all_timestamp': False,
}

CONFIG_FINETUNING = {
    # configs for the models that have to be finetuned on fraud detection task
    'use_embeddings': False,
    'create_equal_parameters_baselines': True,
    'weighted_loss': False,
    'pretrained_model_folder': 'test_training_base',
    'pretrained_model_checkpoint': None,
    'load_weights': True,
    'output_dir': './results_ablation_czech',
    'experiment_name': 'test_lstm1_baseline_equal',

    'model_type': 'lstm',

    'hidden_size': 792,
    'num_layers_lstm': 1,
    'num_layers_mlp': 4,

    'random_seq_start': False,
    'seq_len': 100,
    'batch_size': 20, # old: 15, # 10,
    'epochs': 30, # old: 15, # 30,

    ### parameters for dataset creation when no pretrained model is used
    'val_split_percentage': 0.0,
    'test_split_percentage': 0.20,
    'kfold_nsplits': 5,
    'nrows': None,
    'cached': True,
    'user_ids': None,
    'segno_amount': 1,  # 0 non usa segno, 1 usa campo aggiuntivo, 2 somma il valore assoluto del min
    'skip_user': False,
    'columns_to_select': ["date_trans", "type_trans", "operation", "amount_trans", "balance",
                             "k_symbol",  # transaction
                             'account_id', ],
    ###### DO NOT CHANGE ARE ONLY FOR BASELINE ######
    'flatten': True,
    'regression': False,
    'quantize_regression_output': False,
    'combinatorial_quantization': False,
    'quantize_num_bins': 10,
    'numerical_fields': [],
    'regression_fields': [],
    'combinatorial_quantization_fields': [],
    'numerical_num_bins': None,
    'normalize_regression': None,
    'num_figures': None,
    'combinatorial_quantization_base': None,
    'combinatorial_quantization_comma': None,
    ###### DO NOT CHANGE ARE ONLY FOR BASELINE ######
}


def check_configs():
    if "account_id" in CONFIG_DICT['columns_to_select']:
        CONFIG_DICT['columns_to_select'].remove("account_id")

    ## check that combinatorial quantization parameters are coherent
    if not CONFIG_DICT['combinatorial_quantization']:
        CONFIG_DICT['combinatorial_quantization_fields'] = []
    else:
        CONFIG_DICT['numerical_fields'] = []

    ## check that regression parameters are coherent
    if not CONFIG_DICT['regression']:
        CONFIG_DICT['regression_fields'] = []
    else:
        if not CONFIG_DICT['quantize_regression_output']:
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

    assert len(CONFIG_FINETUNING['numerical_fields']) == 0
    assert len(CONFIG_FINETUNING['regression_fields']) == 0
    assert len(CONFIG_FINETUNING['combinatorial_quantization_fields']) == 0

    # if CONFIG_FINETUNING['create_equal_parameters_baselines']:
    #     assert CONFIG_DICT['field_hidden_size'] == CONFIG_FINETUNING['hidden_size']

    if CONFIG_DICT['use_splitted_timestamp']:
        col_to_select = set(CONFIG_DICT['columns_to_select'])
        col_to_select.discard('date_trans')
        col_to_select.update(['Year', 'Month', 'Day'])
        CONFIG_DICT['columns_to_select'] = list(sorted(col_to_select))
        assert 'date_trans' not in CONFIG_DICT['columns_to_select']
        assert 'Year' in CONFIG_DICT['columns_to_select'] and 'Month' in CONFIG_DICT['columns_to_select']
        assert 'Day' in CONFIG_DICT['columns_to_select']
    else:
        col_to_select = set(CONFIG_DICT['columns_to_select'])
        col_to_select.add('date_trans')
        for el in ['Year', 'Month', 'Day']:
            col_to_select.discard(el)
        CONFIG_DICT['columns_to_select'] = list(sorted(col_to_select))
        assert 'date_trans' in CONFIG_DICT['columns_to_select']
        assert 'Year' not in CONFIG_DICT['columns_to_select'] and 'Month' not in CONFIG_DICT['columns_to_select']
        assert 'Day' not in CONFIG_DICT['columns_to_select']

    if CONFIG_DICT['embedding_type'] == 'field_embedding':
        assert CONFIG_DICT['field_emb_size'] > 0
    else:
        CONFIG_DICT['field_emb_size'] = 0

    num_cols = len(CONFIG_DICT['columns_to_select'])
    if CONFIG_DICT['combinatorial_quantization']:
        for field in CONFIG_DICT['combinatorial_quantization_fields']:
            num_cols += CONFIG_DICT['num_figures'] - 1
    # if not CONFIG_DICT['hidden_size'] == CONFIG_DICT['field_hidden_size'] * num_cols:
    #     logger.debug(
    #         f"{CONFIG_DICT['hidden_size']} should be {num_cols}*{CONFIG_DICT['field_hidden_size']}, \ncolumns to select are {CONFIG_DICT['columns_to_select']}")
    #     logger.debug(f"changing hidden size to {num_cols * CONFIG_DICT['field_hidden_size']}")
    #     CONFIG_DICT['hidden_size'] = CONFIG_DICT['field_hidden_size'] * num_cols
    CONFIG_DICT['num_cols'] = num_cols
