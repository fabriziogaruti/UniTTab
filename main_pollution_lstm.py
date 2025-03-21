from copy import deepcopy
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

from transformers import Trainer, TrainingArguments, DefaultDataCollator
from dataset.prsa import PRSADataset, PRSADatasetEmbedded
from models.baseline_models_prsa import build_baseline_model
from configs.config_prsa import CONFIG_DICT, CONFIG_FINETUNING, check_configs
from testing.test_utils_prsa import test_prsa_model
from dataset.vocab import EncoderVocab
from evaluation.compute_metrics_prsa import PrsaEvaluator
from misc.utils import random_split_dataset
from args import define_ablation_parser, change_ablation_config


def main():
    # set random seeds
    seed = CONFIG_DICT['seed']
    random.seed(seed)  # python
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # torch.cuda

    use_embeddings = CONFIG_FINETUNING['use_embeddings']
    data_root = CONFIG_DICT['data_root']
    output_dir = join(CONFIG_FINETUNING['output_dir'], CONFIG_FINETUNING['experiment_name'])
    equal_parameters_baselines = CONFIG_FINETUNING['create_equal_parameters_baselines']

    pretraining_configs = json.load(open(join(output_dir, "args_pretraining.json"), 'r'))
    dataset_configs = pretraining_configs.copy()
    dataset_configs['seq_len'] = CONFIG_FINETUNING['seq_len']
    dataset_configs['stride'] = CONFIG_FINETUNING['stride']

    raw_dataset = PRSADataset(configs=dataset_configs,
                          root=data_root,
                          vocab_dir=output_dir,
                          return_labels=True,
                          masked_lm=True,
                          partial_ds=False,
                          baseline=False,
                          )

    ### START EXPORTING STEPS
    encoder_vocab = EncoderVocab(
        encoder_fit=None,
        input_vocab=raw_dataset.input_vocab,
        output_vocab=raw_dataset.output_vocab
    )
    with open(os.path.join(output_dir,'encoder_vocab.pkl'), 'wb') as out:
        pickle.dump(encoder_vocab, out, pickle.HIGHEST_PROTOCOL)
    
    script_model = torch.jit.load(os.path.join(output_dir,'model_script.pt'))
    logger.info('succesfully load the model')
    
    dataset = PRSADatasetEmbedded(
                            pretrained_model=script_model,
                            raw_dataset=raw_dataset
                            )
       
    # take parameters from config
    epochs=CONFIG_FINETUNING['epochs']
    batch_size=CONFIG_FINETUNING['batch_size']
    baseline_model_type=CONFIG_FINETUNING['model_type']
    hidden_size = CONFIG_FINETUNING['hidden_size']
    num_layers = CONFIG_FINETUNING['num_layers_lstm'] if baseline_model_type=='lstm' else CONFIG_FINETUNING['num_layers_mlp']
    sequence_len = CONFIG_FINETUNING['seq_len']
    input_size = pretraining_configs['hidden_size']
    field_input_size = pretraining_configs['field_hidden_size']
    vocab_size = dataset.input_vocab.__len__()

    ### define the model
    model = build_baseline_model(baseline_model_type, use_embeddings, hidden_size, sequence_len, num_layers, input_size, field_input_size, vocab_size, equal_parameters_baselines)

    # split the dataset
    totalN = len(dataset)
    valN = int(CONFIG_FINETUNING['val_split_percentage'] * totalN)
    testN = int(CONFIG_FINETUNING['test_split_percentage'] * totalN)

    trainN = totalN - valN - testN
    assert totalN == trainN + valN + testN
    lengths = [trainN, valN, testN]
    logger.info(f"# lengths: train [{trainN}]  valid [{valN}]  test [{testN}]")
    logger.info("# lengths: train [{:.2f}]  valid [{:.2f}]  test [{:.2f}] \n".format(
        trainN / totalN, valN / totalN, testN / totalN))
    train_dataset, eval_dataset, test_dataset = random_split_dataset(dataset, lengths)
    # train_indices, test_indices, eval_indices = dataset.resample_train(train_dataset.indices, test_dataset.indices, eval_dataset.indices)
    # train_dataset.dataset, train_dataset.indices = dataset, train_indices
    # eval_dataset.dataset, eval_dataset.indices = dataset, eval_indices
    # test_dataset.dataset, test_dataset.indices = dataset, test_indices
    if valN == 0:
        eval_dataset = deepcopy(test_dataset)

    # Calc the steps
    save_steps_every = epochs / 20
    calc_steps = max(int(len(train_dataset)/(batch_size*max(1,torch.cuda.device_count()))*save_steps_every), 1)
    logger.info(f"Save steps every {save_steps_every} epochs")
    logger.info(f"Save steps: {calc_steps}")

    # define the evaluator
    scaler_pm10 = dataset.encoding_fn['PM10']
    scaler_pm25 = dataset.encoding_fn['PM2.5']
    evaluator = PrsaEvaluator(scaler_pm10=scaler_pm10, scaler_pm25=scaler_pm25)

    training_args = TrainingArguments(
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,  # def=8 flat=30 seqlen=15
        per_device_eval_batch_size=batch_size,  # def=8 flat=30 seqlen=15
        output_dir=output_dir,
        overwrite_output_dir=True,
        report_to='tensorboard',  # report_to="wandb",

        logging_strategy="steps", logging_steps=100,
        load_best_model_at_end=True,
        save_strategy="steps", save_steps=calc_steps,  # epoch # steps # opts.save_steps
        evaluation_strategy="steps", eval_steps=calc_steps,  # epoch # steps # opts.save_steps
        eval_accumulation_steps=90,
        dataloader_num_workers=4,

        remove_unused_columns=False,
        include_inputs_for_metrics=True,

        metric_for_best_model="Mae_tot",
        greater_is_better=False,
        save_total_limit=2,
    )

    data_collator = DefaultDataCollator()

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=evaluator.compute_metrics_prsa,
    )

    trainer.train()
    # wandb.finish()

    logger.info('### STARTING TESTING ###')

    model.eval()
    with torch.no_grad():
        test_prsa_model(model, test_dataset, output_dir)


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
