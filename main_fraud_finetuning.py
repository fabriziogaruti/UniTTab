import os
import pickle
from os import makedirs
from os.path import join
from loguru import logger
import numpy as np
import torch
import random
import json
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from dataset.card import TransactionDataset
from dataset.datacollator import TransDataCollatorFinetuning

from configs.config_card import CONFIG_DICT, CONFIG_FINETUNING, check_configs
from testing.test_utils_fraud import test_fraud_model_new
from dataset.vocab import EncoderVocab
from misc.utils import random_split_dataset
from args import define_ablation_parser, change_ablation_config
from transformers import BertTokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    # set random seeds
    logger.info(f'STARTING FINETUNING')
    seed = CONFIG_DICT['seed']
    random.seed(seed)  # python
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # torch.cuda
    logger.info(f'setted seed {seed}')

    data_root = CONFIG_DICT['data_root']
    data_fname = CONFIG_DICT['data_fname']
    output_dir = join(CONFIG_FINETUNING['output_dir'], CONFIG_FINETUNING['experiment_name'])

    pretraining_configs = json.load(open(join(output_dir, "args_pretraining.json"), 'r'))
    dataset_configs = pretraining_configs.copy()
    dataset_configs['seq_len'] = CONFIG_FINETUNING['seq_len']
    dataset_configs['nrows'] = CONFIG_FINETUNING['nrows']
    dataset_configs['stride'] = CONFIG_FINETUNING['stride']
    dataset_configs['seed'] = CONFIG_DICT['seed']

    dataset = TransactionDataset(configs=dataset_configs,
                root=data_root,
                fname=data_fname,
                vocab_dir=output_dir,
                return_labels=True,
                masked_lm=True,
                baseline=False,
                )

    ### START EXPORTING STEPS
    encoder_vocab = EncoderVocab(
        encoder_fit=None,
        input_vocab=dataset.input_vocab,
        output_vocab=dataset.output_vocab
    )
    with open(os.path.join(output_dir,'encoder_vocab.pkl'), 'wb') as out:
        pickle.dump(encoder_vocab, out, pickle.HIGHEST_PROTOCOL)
    
    script_model = torch.jit.load(os.path.join(output_dir,'model_script.pt'))
    logger.info('succesfully load the model')
    
    # split the dataset
    totalN = len(dataset)
    valN = int(CONFIG_FINETUNING['val_split_percentage'] * totalN)
    testN = int(CONFIG_FINETUNING['test_split_percentage'] * totalN)
    testN = int(0.10 * totalN)

    trainN = totalN - valN - testN
    assert totalN == trainN + valN + testN
    lengths = [trainN, valN, testN]
    logger.info(f"# lengths: train [{trainN}]  valid [{valN}]  test [{testN}]")
    logger.info("# lengths: train [{:.2f}]  valid [{:.2f}]  test [{:.2f}] \n".format(
        trainN / totalN, valN / totalN, testN / totalN))
    train_dataset, eval_dataset, test_dataset = random_split_dataset(dataset, lengths)
    train_indices, test_indices, eval_indices = dataset.resample_train(train_dataset.indices, test_dataset.indices, eval_dataset.indices)
    train_dataset.dataset, train_dataset.indices = dataset, train_indices
    eval_dataset.dataset, eval_dataset.indices = dataset, eval_indices
    test_dataset.dataset, test_dataset.indices = dataset, test_indices
    if valN == 0:
        eval_dataset = test_dataset

    vocab_file = dataset.input_vocab.filename
    special_tokens = dataset.input_vocab.get_special_tokens()
    tokenizer = BertTokenizer(vocab_file,
                                    do_lower_case=False,
                                unk_token=special_tokens.unk_token,
                                pad_token=special_tokens.pad_token,
                                cls_token=special_tokens.cls_token,
                                mask_token=special_tokens.mask_token,
                                bos_token=special_tokens.bos_token,
                                eos_token=special_tokens.eos_token,
                                    )
    data_collator = TransDataCollatorFinetuning(tokenizer=tokenizer, seq_len=CONFIG_DICT['seq_len'])

    logger.info('### STARTING TESTING ###')
    script_model = script_model.to(device)
    script_model.eval()

    with torch.no_grad():
        test_fraud_model_new(script_model, test_dataset, output_dir, data_collator)


if __name__ == "__main__":

    parser = define_ablation_parser()
    args = parser.parse_args()
    CONFIG_DICT, CONFIG_FINETUNING = change_ablation_config(args, CONFIG_DICT, CONFIG_FINETUNING)
    check_configs()

    all_parameters = {
        'general_configuration_file': CONFIG_DICT,
        'finetuning_configuration_file': CONFIG_FINETUNING,
        'command_line_args': vars(args),
    }

    output_dir = join(CONFIG_FINETUNING['output_dir'], CONFIG_FINETUNING['experiment_name'])
    args_dir = join(output_dir, "args_train.json")
    log_dir = join(output_dir, "logs")
    logging_file = join(log_dir, 'loguru.txt')
    makedirs(output_dir, exist_ok=True)
    makedirs(log_dir, exist_ok=True)
    logger.add(logging_file, mode='w')
    with open(args_dir, 'w') as f:
        json.dump(all_parameters, f, indent=6)

    main()
