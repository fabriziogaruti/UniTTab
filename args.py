import argparse
from loguru import logger

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def define_ablation_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    # general arguments
    parser.add_argument("--data_fname", type=str,
                        default=None,
                        help='file name of transaction')
    parser.add_argument("--pretraining_output_dir", type=str,
                        default=None,
                        help='directory containing all pretraining results')
    parser.add_argument("--finetuning_output_dir", type=str,
                        default=None,
                        help='directory containing all finetuning results')
    parser.add_argument("--pretraining_experiment_name", type=str,
                        default=None,
                        help='folder containing this pretraining results')
    parser.add_argument("--finetuning_experiment_name", type=str,
                        default=None,
                        help='folder containing this finetuning results')
    parser.add_argument("--pretraining_exp_to_load", type=str,
                        default=None,
                        help='folder containing the pretraining results to load for finetuning')
    parser.add_argument("--pretraining_epochs", type=int,
                        default=None,
                        help='number of epochs for pretraining')
    parser.add_argument("--finetuning_epochs", type=int,
                        default=None,
                        help='number of epochs for pretraining')
    parser.add_argument("--seq_len", type=int,
                        default=None,
                        help='seq_len')
    parser.add_argument("--stride", type=int,
                        default=None,
                        help='stride')
    parser.add_argument("--seed", type=int,
                        default=None,
                        help='seed')
    parser.add_argument("--load_weights", type=str2bool,
                        default=None,
                        help='wether to load weights or train from scratch')
    parser.add_argument("--percentage_pretraining", type=float,
                        default=None,
                        help='percentage of pretraining data to use')
        
    ### arguments on handling of numerical fields
    parser.add_argument("--segno_amount", type=int,
                        default=None,
                        help="number to decide how to handle negative amounts: 0 non usa segno, 1 usa campo aggiuntivo, 2 somma il valore assoluto del min")
    parser.add_argument("--use_numerical_fields", type=str2bool,
                        default=None,
                        help="flag to decide whether to use more bins for numerical fields")
    parser.add_argument("--regression", type=str2bool,
                        default=None,
                        help="flag to decide whether to use regression")
    parser.add_argument("--combinatorial_quantization", type=str2bool,
                        default=None,
                        help="flag to decide whether to use combinatorial quantization")
    parser.add_argument("--regression_input_type", type=str,
                        default=None,
                        help="string to decide whether to use 'linear' of 'positional_embedding' to encode numerical data")
    parser.add_argument("--d_pos_enc", type=int,
                    default=None,
                    help="dimension of the positional encoding embedding for numerical data")
    parser.add_argument("--quantize_regression_output", type=str2bool,
                        default=None,
                        help="flag to decide whether to use quantize loss for regression numerical data")

    ### arguments on loss functions
    parser.add_argument("--categorical_label_smoothing", type=str2bool,
                        default=None,
                        help="flag to decide whether to use label smoothing on categorical data")
    parser.add_argument("--numerical_label_smoothing", type=str2bool,
                        default=None,
                        help="flag to decide whether to use custom label smoothing with radius on numerical data")
    parser.add_argument("--regression_loss_function", type=str,
                        default=None,
                        help="string that determines regression loss function (can be  L1, L2, L1smooth)")

    ### arguments on timestamp encoding
    parser.add_argument("--use_splitted_timestamp", type=str2bool,
                        default=None,
                        help="flag to decide whether to encode timestamp splitting in year, month, day, hour")
    parser.add_argument("--consider_numerical_timestamp", type=str2bool,
                        default=None,
                        help="flag to decide whether to encode timestamp splitting in the numerical way")

    ### argument to substitute positional encoding with field embedding
    ### possible values: 'no': standard senza niente
    ###                  'positional_encoding': add positional encoding
    ###                  'field_embedding': no pos_enc ma field embedding
    parser.add_argument("--embedding_type", type=str,
                    default=None,
                    help="string to decide whether to use field embedding, positional encoding or nothing in the field transformer")
    parser.add_argument("--field_emb_size", type=int,
                    default=None,
                    help="dimension of field embedding, if used")

    ### arguments on type of baseline
    parser.add_argument("--use_embeddings", type=str2bool,
                        default=None,
                        help="flag to decide whether to use the version of baselines with equal parameters")
    parser.add_argument("--equal_parameters_baselines", type=str2bool,
                        default=None,
                        help="flag to decide whether to use the version of baselines with equal parameters")
    ### arguments on xgboost feature creation
    parser.add_argument("--use_minimal_feature", type=str2bool,
                        default=None,
                        help="flag to decide whether to use the version of baselines with equal parameters")
    parser.add_argument("--use_catboost", type=str2bool,
                        default=None,
                        help="flag to decide whether to use catboost or xgboost")
    parser.add_argument("--aggregate_on_time", type=str2bool,
                        default=None,
                        help="flag to decide whether to use tsfresh")
    parser.add_argument("--use_categorical", type=str2bool,
                        default=None,
                        help="flag to decide whether to use the version of baselines with equal parameters")
    parser.add_argument("--use_onehot", type=str2bool,
                        default=None,
                        help="flag to decide whether to use the version of baselines with equal parameters")


    ### arguments on model sizes
    parser.add_argument("--hidden_size", type=int,
                    default=None,
                    help="hidden size of tabbert model")
    parser.add_argument("--field_hidden_size", type=int,
                    default=None,
                    help="field hidden size of tabbert model")  
    parser.add_argument("--lstm_hidden_size", type=int,
                    default=None,
                    help="hidden size of baseline lstm model")  
    parser.add_argument("--num_layers_lstm", type=int,
                    default=None,
                    help="num layers of baseline lstm model")

    ### arguments on masking methods
    parser.add_argument("--mlm_probability", type=float,
                        default=None,
                        help="float to change the values of masking element prob")
    parser.add_argument("--mask_line", type=str2bool,
                        default=None,
                        help="flag to decide whether to use masking of entire lines")
    parser.add_argument("--mask_line_probability", type=float,
                        default=None,
                        help="float to change the values of masking line prob")
    parser.add_argument("--mask_with_specific_dict", type=str2bool,
                        default=None,
                        help="flag to decide whether to use specific dictionary in masking for random words")
    parser.add_argument("--mask_all_timestamp", type=str2bool,
                        default=None,
                        help="flag to decide whether to mask all the timestamp fields together when splitted")
    parser.add_argument("--batch_size", type=int,
                        default=None,
                        help="batch_size")

    parser.add_argument("--finetuning_test_split_percentage", type=float,
                        default=None,
                        help='test split percentage of the dataset (the same subset will be used for validation)')
    parser.add_argument("--weighted_loss", type=str2bool,
                        default=None,
                        help="flag to decide whether to use have a weighted loss in binary classification")
    return parser

def change_ablation_config(args, config_pretraining, config_finetuning):

    if args.data_fname is not None:
        config_pretraining['data_fname'] = args.data_fname        
    if args.pretraining_output_dir is not None:
        config_pretraining['output_dir'] = args.pretraining_output_dir
    if args.finetuning_output_dir is not None:
        config_finetuning['output_dir'] = args.finetuning_output_dir 
    if args.pretraining_experiment_name is not None:
        config_pretraining['experiment_name'] = args.pretraining_experiment_name 
    if args.finetuning_experiment_name is not None:
        config_finetuning['experiment_name'] = args.finetuning_experiment_name 
    if args.pretraining_exp_to_load is not None:
        config_finetuning['pretrained_model_folder'] = args.pretraining_exp_to_load 
    if args.pretraining_epochs is not None:
        config_pretraining['epochs'] = args.pretraining_epochs
    if args.finetuning_epochs is not None:
        config_finetuning['epochs'] = args.finetuning_epochs
    if args.seq_len is not None:
        config_pretraining['seq_len'] = args.seq_len
        config_finetuning['seq_len'] = args.seq_len
    if args.stride is not None:
        config_pretraining['stride'] = args.stride
        config_finetuning['stride'] = args.stride
    if args.seed is not None:
        config_pretraining['seed'] = args.seed
    if args.load_weights is not None:
        config_finetuning['load_weights'] = args.load_weights
    if args.percentage_pretraining is not None:
        config_pretraining['percentage_pretraining'] = args.percentage_pretraining
        
    if args.segno_amount is not None:
        config_pretraining['segno_amount'] = args.segno_amount
        config_finetuning['segno_amount'] = args.segno_amount
    if args.use_numerical_fields is not None:
        if not args.use_numerical_fields:
            config_pretraining['numerical_fields'] = []
    if args.regression is not None:
        config_pretraining['regression'] = args.regression
    if args.combinatorial_quantization is not None:
        config_pretraining['combinatorial_quantization'] = args.combinatorial_quantization
    if args.regression_input_type is not None:
        config_pretraining['regression_input_type'] = args.regression_input_type
    if args.d_pos_enc is not None:
        config_pretraining['d_pos_enc'] = args.d_pos_enc
    if args.quantize_regression_output is not None:
        config_pretraining['quantize_regression_output'] = args.quantize_regression_output        

    # if args.numerical_with_pos_emb is not None:
    #     config_pretraining['numerical_with_pos_emb'] = args.numerical_with_pos_emb

    if args.categorical_label_smoothing is not None:
        config_pretraining['categorical_label_smoothing'] = args.categorical_label_smoothing
    if args.numerical_label_smoothing is not None:
        config_pretraining['numerical_label_smoothing'] = args.numerical_label_smoothing
    if args.regression_loss_function is not None:
        config_pretraining['loss_regression'] = args.regression_loss_function

    if args.use_splitted_timestamp is not None:
        config_pretraining['use_splitted_timestamp'] = args.use_splitted_timestamp
    if args.consider_numerical_timestamp is not None:
        config_pretraining['consider_numerical_timestamp'] = args.consider_numerical_timestamp

    if args.embedding_type is not None:
        config_pretraining['embedding_type'] = args.embedding_type
    if args.field_emb_size is not None:
        config_pretraining['field_emb_size'] = args.field_emb_size
        
    if args.hidden_size is not None:
        config_pretraining['hidden_size'] = args.hidden_size
    if args.field_hidden_size is not None:
        config_pretraining['field_hidden_size'] = args.field_hidden_size

    if args.use_embeddings is not None:
        config_finetuning['use_embeddings'] = args.use_embeddings
    if args.equal_parameters_baselines is not None:
        config_finetuning['create_equal_parameters_baselines'] = args.equal_parameters_baselines
    if args.lstm_hidden_size is not None:
        config_finetuning['hidden_size'] = args.lstm_hidden_size
    if args.num_layers_lstm is not None:
        config_finetuning['num_layers_lstm'] = args.num_layers_lstm

    if args.mlm_probability is not None:
        config_pretraining['mlm_prob'] = args.mlm_probability
    if args.mask_line is not None:
        config_pretraining['mask_line'] = args.mask_line
    if args.mask_line_probability is not None:
        config_pretraining['mask_line_probability'] = args.mask_line_probability
    if args.mask_with_specific_dict is not None:
        config_pretraining['mask_specific_dict'] = args.mask_with_specific_dict
    if args.mask_all_timestamp is not None:
        config_pretraining['mask_all_timestamp'] = args.mask_all_timestamp

    if args.use_minimal_feature is not None:
        config_finetuning['use_minimal_feature'] = args.use_minimal_feature
    if args.use_catboost is not None:
        config_finetuning['use_catboost'] = args.use_catboost
    if args.aggregate_on_time is not None:
        config_finetuning['aggregate_on_time'] = args.aggregate_on_time
    if args.use_categorical is not None:
        config_finetuning['use_categorical'] = args.use_categorical
    if args.use_onehot is not None:
        config_finetuning['use_onehot'] = args.use_onehot

    if args.batch_size is not None:
        config_pretraining['batch_size'] = args.batch_size
        config_finetuning['batch_size'] = args.batch_size

    if args.finetuning_test_split_percentage is not None:
        config_finetuning['test_split_percentage'] = args.finetuning_test_split_percentage

    if args.weighted_loss is not None:
        config_finetuning['weighted_loss'] = args.weighted_loss

    return config_pretraining, config_finetuning